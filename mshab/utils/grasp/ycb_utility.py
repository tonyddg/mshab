from typing import List, Optional, Tuple, Union
import sapien
import numpy as np

import torch
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
from mani_skill.utils.structs.pose import Pose

from mani_skill import ASSET_DIR
REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"

# 异形物体以及对应的预设抓取位姿 (mshab)
SPECIAL_OBJECT_GRASP_POSE_DICT = {
    "024_bowl": {
        "path": REARRANGE_DIR / "grasp_poses/set_table/024_bowl/grasp_poses.pt",
        # 是否绕物体自身 z 轴旋转对称
        "z_axis_rot_symmetry": True,
        # 夹取该物体时的总夹爪宽度（两指间距）
        "gripper_width": 0.02,
    }
}

def get_special_object_grasp_pose_cfg(id: str):
    cfg = SPECIAL_OBJECT_GRASP_POSE_DICT[id]
    return {
        "path": cfg["path"],
        "z_axis_rot_symmetry": bool(cfg.get("z_axis_rot_symmetry", False)),
        "gripper_width": cfg.get("gripper_width", None),
    }

# 读取异形物体抓取位姿
def sample_special_object_grasp_pose(
    id: str, num_sample: int
):
    '''
    返回值为 obj_raw_pose_wrt_tcp, tcp 观测下物体的位姿
    pose 格式固定为 [x, y, z, qw, qx, qy, qz]
    '''
    cfg = get_special_object_grasp_pose_cfg(id)

    success_obj_raw_pose_wrt_tcp = torch.load(
        cfg["path"], # type: ignore
        map_location="cuda"
    )["success_obj_raw_pose_wrt_tcp"]

    sample_idx = torch.randint(
        low=0,
        high=success_obj_raw_pose_wrt_tcp.shape[0],
        size=(num_sample,),
        dtype=torch.long, 
        device="cpu"
    )
    sampled_obj_raw_pose_wrt_tcp = torch.as_tensor(
        success_obj_raw_pose_wrt_tcp[sample_idx],
        dtype=torch.float32, 
        device="cpu"
    ).reshape(-1, 7)

    return sampled_obj_raw_pose_wrt_tcp

def build_special_object_grasp_T(
    id: str,
    obj_origin_T: torch.Tensor,
):
    '''
    输入:
        obj_origin_T: [N, 4, 4]
            物体原始位姿的齐次矩阵（相对于当前参考系）
            - 在 MobileReachObject 中它是 T_world_obj
            - 在 MobileReachPlace 中它是 T_obj_obj = I，因此输出就是 T_obj_tcp

    返回:
        special_grasp_T: [N, 4, 4]
            与 obj_origin_T 同一参考系下的 TCP 抓取位姿
    '''
    cfg = get_special_object_grasp_pose_cfg(id)

    obj_origin_T = torch.as_tensor(
        obj_origin_T,
        dtype=torch.float32
    ).reshape(-1, 4, 4)

    device = obj_origin_T.device
    num_sample = obj_origin_T.shape[0]

    sampled_obj_raw_pose_wrt_tcp = sample_special_object_grasp_pose(id, num_sample).to(device)
    sampled_T_tcp_obj = torch.as_tensor(
        Pose.create_from_pq(
            p=sampled_obj_raw_pose_wrt_tcp[:, :3],
            q=sampled_obj_raw_pose_wrt_tcp[:, 3:]
        ).to_transformation_matrix(),
        dtype=torch.float32,
        device=device
    )

    # 候选 1：原始采样
    # T_ref_tcp = T_ref_obj @ T_obj_tcp = T_ref_obj @ inv(T_tcp_obj)
    special_grasp_T = obj_origin_T @ torch.linalg.inv(sampled_T_tcp_obj)

    # 若物体绕自身 z 轴旋转对称，则构造候选 2 并比较谁的 x 轴更接近世界 +x
    if cfg["z_axis_rot_symmetry"]:
        rot_z_pi = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_sample, 1, 1)
        rot_z_pi[:, 0, 0] = -1.0
        rot_z_pi[:, 1, 1] = -1.0

        sampled_T_tcp_obj_flip = sampled_T_tcp_obj @ rot_z_pi
        special_grasp_T_flip = obj_origin_T @ torch.linalg.inv(sampled_T_tcp_obj_flip)

        # 比较 special_grasp_T 的 x 轴与世界 +x 的接近程度
        # 旋转矩阵第 1 列是 x 轴方向，因此直接看 [0, 0]
        score = special_grasp_T[:, 0, 0]
        score_flip = special_grasp_T_flip[:, 0, 0]
        use_flip_mask = score_flip > score

        if use_flip_mask.any():
            special_grasp_T[use_flip_mask] = special_grasp_T_flip[use_flip_mask]

    return special_grasp_T

YCB_DATASET = dict()
def _load_ycb_dataset():
    global YCB_DATASET
    try:
        YCB_DATASET = {
            "model_data": load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"),
        }
    except:
        raise RuntimeError("需要下载 ycb 数据集: python -m mani_skill.utils.download_asset 'ycb'")

def _query_ycb_dataset(id: str):
    if "YCB" not in YCB_DATASET:
        _load_ycb_dataset()
    model_db = YCB_DATASET["model_data"]
    return dict(model_db[id])

def get_ycb_size(id: str) -> np.ndarray:
    metadata = _query_ycb_dataset(id)
    model_scales = metadata.get("scales", [1.0])
    scale = model_scales[0]
    bbox_max = metadata["bbox"]["max"]
    bbox_min = metadata["bbox"]["min"]

    model_size = np.array((
        float((bbox_max[0] - bbox_min[0]) * scale),
        float((bbox_max[1] - bbox_min[1]) * scale),
        float((bbox_max[2] - bbox_min[2]) * scale)
    ))
    return model_size

def get_ycb_builder(
    scene: Union[ManiSkillScene, sapien.Scene], id: str, add_collision: bool = True, add_visual: bool = True
):
    builder = scene.create_actor_builder()

    metadata = _query_ycb_dataset(id)
    density = metadata.get("density", 1000)
    model_scales = metadata.get("scales", [1.0])
    scale = model_scales[0]

    model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / id
    if add_collision:
        collision_file = str(model_dir / "collision.ply")
        builder.add_multiple_convex_collisions_from_file(
            filename = collision_file,
            scale = [scale] * 3, # type: ignore
            material = None,
            density = density,
        )
    if add_visual:
        visual_file = str(model_dir / "textured.obj")
        builder.add_visual_from_file(
            filename = visual_file, 
            scale = [scale] * 3 # type: ignore
        )

    return builder
