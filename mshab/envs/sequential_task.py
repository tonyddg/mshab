import copy
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import torch
import torch.random
import transforms3d
from transforms3d.euler import euler2quat

import sapien

from mani_skill.agents.robots import Fetch
from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_apply,
    quaternion_invert,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Articulation, Pose
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import vectorize_pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from mshab.envs.planner import (
    ArticulationConfig,
    CloseSubtask,
    CloseSubtaskConfig,
    NavigateSubtask,
    NavigateSubtaskConfig,
    OpenSubtask,
    OpenSubtaskConfig,
    PickSubtask,
    PickSubtaskConfig,
    PlaceSubtask,
    PlaceSubtaskConfig,
    Subtask,
    SubtaskConfig,
    TaskPlan,
)
from mshab.utils.array import (
    all_equal,
    all_same_type,
    tensor_intersection,
    tensor_intersection_idx,
)

from mani_skill.utils.geometry import rotation_conversions

from mshab.utils.grasp.compute_grasp_pose import compute_grasp_pose_by_obb_torch
from mshab.utils.grasp.ycb_utility import get_ycb_size, build_special_object_grasp_T, SPECIAL_OBJECT_GRASP_POSE_DICT, get_special_object_grasp_pose_cfg

# 末端连杆名称
FETCH_TCP_LINK_NAME = "gripper_link"
# 移动底盘所在连杆名称
FETCH_BASE_LINK_NAME = "base_link"
# 夹爪最大行程
FETCH_GRIPPER_OPEN_QPOS = 0.050
# 夹爪关节在 Sapien 标准 QPOS 中的索引
FETCH_GRIPPER_QPOS_IDX = [13, 14]
# 关节夹爪在 Fetch agent 中的索引
FETCH_GRIPPER_ACT_IDX = 7
# 与原版的 Fetch 存在不同, 可能需要同步到原版的 Fetch
FETCH_HEAD_CAMERA_LINK = "head_camera_link"
FETCH_GRIPPER_CAMERA_LINK = "gripper_link"
FETCH_FORWARD_IN_BASE_LINK: np.ndarray = np.array([1.0, 0])

UNIQUE_SUCCESS_SUBTASK_TYPE = 100
GOAL_POSE_Q = transforms3d.quaternions.axangle2quat(
    np.array([0, 1, 0]), theta=np.deg2rad(90)
)

##### 辅助函数 #####

def quaternion_to_rpy_eular(quat: torch.Tensor):
    # 转为 RPY 角
    rot_matrix = rotation_conversions.quaternion_to_matrix(quat)
    # 固定坐标系下的 XYZ 欧拉角 (matrix_to_euler_angles 使用的是运动坐标系, 要传 ZYX 再反序)
    rot_rpy = rotation_conversions.matrix_to_euler_angles(rot_matrix, "ZYX")
    rot_rpy = torch.flip(rot_rpy, dims = [1, ])
    return rot_rpy

# 使用与 reach 环境相同的四元数轴角对转化函数
# 该函数与 maniskill 提供的 rotation_conversions.quaternion_to_axis_angle 存在区别
def quat_wxyz_tensor_to_axis_angle(q: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q: (..., 4) 四元数张量，顺序为 (w, x, y, z)
    返回: (..., 3) 轴角向量表示与旋转角 (rotation vector): 方向=旋转轴，模长=旋转角(弧度)
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected shape (..., 4), got {q.shape}")

    # 归一化
    q = q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))

    w = q[..., 0]
    v = q[..., 1:]  # (x, y, z)
    v_norm = v.norm(dim=-1)  # ||v||

    # 为了得到更“短”的角度（0..pi），可将 w<0 的四元数整体取负（等价旋转）
    sign = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w))
    w = w * sign
    v = v * sign.unsqueeze(-1)
    v_norm = v_norm  # v_norm 不变

    # angle = 2 * atan2(||v||, w)
    angle = 2.0 * torch.atan2(v_norm, w.clamp_min(eps))  # (...,)
    # axis = v / ||v||，并输出 rotvec = axis * angle
    axis = v / v_norm.clamp_min(eps).unsqueeze(-1)
    rotvec = axis * angle.unsqueeze(-1)  # (..., 3)

    # 小角度时数值更稳定：q ≈ [1, r/2] => rotvec ≈ 2*v（小角度可能突变为 2pi）
    small = v_norm < 1e-3
    rotvec = torch.where(small.unsqueeze(-1), 2.0 * v, rotvec)
    # angle = torch.where(small.unsqueeze(-1), 2.0 * v_norm, angle).squeeze(-1)
    # rotvec[small] = 2.0 * v
    # angle[small] = 2.0 * v_norm

    return rotvec, angle

# PI0_ACT_ROT_TYPE = Literal["axis_angle", "rot_6d", "rpy_euler"]
# PI0_OBS_ROT_TYPE = Literal["axis_angle", "rot_6d", "quat"]
ROT_TYPE = Literal["axis_angle", "rot_6d", "quat", "rpy_euler"]

def pose_to_target_type(pose: Pose, rot_type: ROT_TYPE, addition_state: List = []):
    rot = pose.get_q()
    if rot_type == "axis_angle":
        rot, _ = quat_wxyz_tensor_to_axis_angle(rot)
    elif rot_type == "rot_6d":
        rot = rotation_conversions.quaternion_to_matrix(rot)
        rot = rotation_conversions.matrix_to_rotation_6d(rot)
    elif rot_type == "rpy_euler":
        rot = quaternion_to_rpy_eular(rot)
    elif rot_type == "quat":
        pass
    else:
        raise NotImplemented(f"Unknown rot_type: {rot_type}")

    return torch.concat(
        [pose.get_p(), rot] + addition_state, dim = 1
    )

VIS_AXIS_RADIUS = 0.01
VIS_AXIS_LENGTH = 0.05

from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.structs.actor import Actor

def make_vis_axis(
    axis_builder: ActorBuilder
):
    axis_builder.add_cylinder_visual(
        radius = VIS_AXIS_RADIUS,
        half_length = VIS_AXIS_LENGTH,
        material = (1, 0, 0),
        pose = sapien.Pose(p = [VIS_AXIS_LENGTH, 0, 0])
    )
    axis_builder.add_cylinder_visual(
        radius = VIS_AXIS_RADIUS,
        half_length = VIS_AXIS_LENGTH,
        material = (0, 1, 0),
        pose = sapien.Pose(p = [0, VIS_AXIS_LENGTH, 0], q = np.asarray(euler2quat(0, 0, np.deg2rad(-90)), dtype = np.float32))
    )
    axis_builder.add_cylinder_visual(
        radius = VIS_AXIS_RADIUS,
        half_length = VIS_AXIS_LENGTH,
        material = (0, 0, 1),
        pose = sapien.Pose(p = [0, 0, VIS_AXIS_LENGTH], q = np.asarray(euler2quat(0, np.deg2rad(90), 0), dtype = np.float32))
    )
    return axis_builder

def get_grasp_force_angle(agent, object: Actor):
    """获取最小夹持力与最大夹持角度 (越接近 0 越好, 说明接触力与接触方向平行)

    Args:
        object (Actor): The object to check if the robot is grasping
    """
    l_contact_forces = agent.scene.get_pairwise_contact_forces(
        agent.finger1_link, object
    )
    r_contact_forces = agent.scene.get_pairwise_contact_forces(
        agent.finger2_link, object
    )
    lforce = torch.linalg.norm(l_contact_forces, axis=1)
    rforce = torch.linalg.norm(r_contact_forces, axis=1)

    # direction to open the gripper
    ldirection = -agent.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
    rdirection = agent.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
    langle = common.compute_angle_between(ldirection, l_contact_forces)
    rangle = common.compute_angle_between(rdirection, r_contact_forces)
    return torch.minimum(lforce, rforce), torch.maximum(langle, rangle)

##### 辅助函数 #####

# TODO: 测试放置的情况
# TODO: 生成抓取位姿时, 当 TCP 相对 YCB 中心过低时要向上偏置

@register_env("SequentialTask-v0")
class SequentialTaskEnv(SceneManipulationEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------

    Success Conditions
    ------------------

    Visualization: link to a video/gif of the task being solved
    """

    # =========================
    # Pick grasp pose settings
    # =========================
    PICK_GRASP_APPROACHING = (0.0, 0.0, -1.0)
    PICK_GRASP_TARGET_CLOSING = None
    PICK_GRASP_DEPTH = 0.0
    PICK_GRASP_ORTHO = True

    SPECIAL_GRASP_CACHE_DEVICE = "cpu"

    def _ensure_pick_grasp_caches(self):
        if not hasattr(self, "_special_grasp_cfg_cache"):
            self._special_grasp_cfg_cache = dict()
        if not hasattr(self, "_special_grasp_obj_raw_pose_cache"):
            self._special_grasp_obj_raw_pose_cache = dict()
        if not hasattr(self, "_ycb_size_cache"):
            self._ycb_size_cache = dict()

    def _split_obj_instance_name(self, obj_id: str) -> str:
        # "024_bowl-0" -> "024_bowl"
        # "003_cracker_box-1" -> "003_cracker_box"
        return obj_id.rsplit("-", 1)[0]

    def _infer_pick_obj_id_for_env(
        self,
        subtask: PickSubtask,
        env_id: int,
        target_obj: Actor,
    ) -> str:
        # 首选 merge 时保留下来的原始 obj_id
        if getattr(subtask, "source_obj_ids", None) is not None:
            return subtask.source_obj_ids[env_id]

        # 兜底：尝试从底层 entity 名称恢复
        if hasattr(target_obj, "_scene_idxs") and hasattr(target_obj, "_objs"):
            local_idx = target_obj._scene_idxs.tolist().index(env_id)
            entity_name = target_obj._objs[local_idx].name
            prefix = f"env-{env_id}_"
            if entity_name.startswith(prefix):
                entity_name = entity_name[len(prefix):]
            return entity_name

        return subtask.obj_id

    def _pose_matrices_to_vec7(self, T: torch.Tensor) -> torch.Tensor:
        """
        T: [B, 4, 4] -> [B, 7]
        输出格式与 vectorize_pose 一致：p(3) + q(4)
        """
        if T.ndim == 2:
            T = T.unsqueeze(0)
        p = T[:, :3, 3]
        q = rotation_conversions.matrix_to_quaternion(T[:, :3, :3])
        return torch.cat([p, q], dim=-1)

    def _is_special_grasp_object(self, ycb_id: str) -> bool:
        self._ensure_pick_grasp_caches()
        if ycb_id in self._special_grasp_cfg_cache:
            return True
        try:
            cfg = get_special_object_grasp_pose_cfg(ycb_id)
        except Exception:
            return False
        self._special_grasp_cfg_cache[ycb_id] = cfg
        return True

    def _get_special_grasp_cfg_cached(self, ycb_id: str) -> Dict[str, Any]:
        self._ensure_pick_grasp_caches()
        if ycb_id not in self._special_grasp_cfg_cache:
            self._special_grasp_cfg_cache[ycb_id] = get_special_object_grasp_pose_cfg(ycb_id)
        return self._special_grasp_cfg_cache[ycb_id]

    def _get_ycb_size_tensor_cached(
        self,
        ycb_id: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self._ensure_pick_grasp_caches()
        if ycb_id not in self._ycb_size_cache:
            self._ycb_size_cache[ycb_id] = torch.as_tensor(
                get_ycb_size(ycb_id),
                dtype=torch.float32,
                device="cpu",
            )
        return self._ycb_size_cache[ycb_id].to(device=device, dtype=dtype)

    def _get_special_obj_raw_pose_wrt_tcp_cache(self, ycb_id: str) -> torch.Tensor:
        """
        返回缓存的 success_obj_raw_pose_wrt_tcp，shape [N, 7]，保存在 CPU。
        """
        self._ensure_pick_grasp_caches()
        if ycb_id not in self._special_grasp_obj_raw_pose_cache:
            cfg = self._get_special_grasp_cfg_cached(ycb_id)
            data = torch.load(
                cfg["path"],
                map_location=self.SPECIAL_GRASP_CACHE_DEVICE,
            )["success_obj_raw_pose_wrt_tcp"]
            self._special_grasp_obj_raw_pose_cache[ycb_id] = torch.as_tensor(
                data,
                dtype=torch.float32,
                device=self.SPECIAL_GRASP_CACHE_DEVICE,
            ).reshape(-1, 7)
        return self._special_grasp_obj_raw_pose_cache[ycb_id]

    def _sample_special_object_grasp_pose_cached(
        self,
        ycb_id: str,
        num_sample: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        返回 sampled_obj_raw_pose_wrt_tcp, shape [num_sample, 7]
        """
        cached = self._get_special_obj_raw_pose_wrt_tcp_cache(ycb_id)  # CPU [N,7]
        sample_idx = torch.randint(
            low=0,
            high=cached.shape[0],
            size=(num_sample,),
            dtype=torch.long,
            device="cpu",
        )
        sampled = cached[sample_idx]
        return sampled.to(device=device, dtype=dtype)

    def _build_special_object_grasp_T_cached(
        self,
        ycb_id: str,
        obj_origin_T: torch.Tensor,
    ) -> torch.Tensor:
        """
        obj_origin_T: [B,4,4]
        返回: [B,4,4]
        逻辑与 build_special_object_grasp_T 一致，但把抓取姿态库做了缓存。
        """
        cfg = self._get_special_grasp_cfg_cached(ycb_id)

        obj_origin_T = torch.as_tensor(
            obj_origin_T,
            dtype=torch.float32,
            device=obj_origin_T.device,
        ).reshape(-1, 4, 4)

        device = obj_origin_T.device
        dtype = obj_origin_T.dtype
        B = obj_origin_T.shape[0]

        sampled_obj_raw_pose_wrt_tcp = self._sample_special_object_grasp_pose_cached(
            ycb_id=ycb_id,
            num_sample=B,
            device=device,
            dtype=dtype,
        )

        sampled_T_tcp_obj = torch.as_tensor(
            Pose.create_from_pq(
                p=sampled_obj_raw_pose_wrt_tcp[:, :3],
                q=sampled_obj_raw_pose_wrt_tcp[:, 3:],
            ).to_transformation_matrix(),
            dtype=dtype,
            device=device,
        )

        # T_ref_tcp = T_ref_obj @ inv(T_tcp_obj)
        special_grasp_T = obj_origin_T @ torch.linalg.inv(sampled_T_tcp_obj)

        if cfg["z_axis_rot_symmetry"]:
            rot_z_pi = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
            rot_z_pi[:, 0, 0] = -1.0
            rot_z_pi[:, 1, 1] = -1.0

            sampled_T_tcp_obj_flip = sampled_T_tcp_obj @ rot_z_pi
            special_grasp_T_flip = obj_origin_T @ torch.linalg.inv(sampled_T_tcp_obj_flip)

            # 选 x 轴更接近世界 +x 的候选
            score = special_grasp_T[:, 0, 0]
            score_flip = special_grasp_T_flip[:, 0, 0]
            use_flip_mask = score_flip > score
            if use_flip_mask.any():
                special_grasp_T[use_flip_mask] = special_grasp_T_flip[use_flip_mask]

        return special_grasp_T

    def _get_pick_grasp_pose_world_for_subtask(
        self,
        subtask_num: int,
        env_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回当前 pick subtask 在 env_idx 上的目标抓取位姿（世界坐标系）
        输出 shape: [len(env_idx), 7]
        """
        subtask = self.task_plan[subtask_num]
        assert isinstance(subtask, PickSubtask)

        target_obj = self.subtask_objs[subtask_num]
        if target_obj is None:
            raise RuntimeError(f"PickSubtask at index {subtask_num} has no target object")

        # Sequential merge 后，pick 目标对象应覆盖全部环境
        if len(target_obj._scene_idxs) != self.num_envs:
            raise NotImplementedError(
                "Pick grasp pose generation currently expects merged pick actor "
                "to cover all environments."
            )

        obj_pose_T_all = target_obj.pose.to_transformation_matrix()  # [num_envs, 4, 4]
        out_pose_vec7 = torch.zeros(
            (env_idx.numel(), 7),
            device=self.device,
            dtype=torch.float32,
        )

        # 先按 ycb_id 分组，便于 batch 计算
        ycb_id_to_local_positions = defaultdict(list)
        for local_i, env_id in enumerate(env_idx.tolist()):
            obj_id = self._infer_pick_obj_id_for_env(subtask, env_id, target_obj)
            ycb_id = self._split_obj_instance_name(obj_id)
            ycb_id_to_local_positions[ycb_id].append(local_i)

        for ycb_id, local_positions in ycb_id_to_local_positions.items():
            local_pos_tensor = torch.as_tensor(
                local_positions, device=env_idx.device, dtype=torch.long
            )
            batch_env_ids = env_idx[local_pos_tensor]
            batch_obj_pose_T = obj_pose_T_all[batch_env_ids]

            if self._is_special_grasp_object(ycb_id):
                grasp_T = self._build_special_object_grasp_T_cached(
                    ycb_id=ycb_id,
                    obj_origin_T=batch_obj_pose_T,
                )
            else:
                size = self._get_ycb_size_tensor_cached(
                    ycb_id=ycb_id,
                    device=batch_obj_pose_T.device,
                    dtype=batch_obj_pose_T.dtype,
                ).view(1, 3).repeat(batch_obj_pose_T.shape[0], 1)

                grasp_T = compute_grasp_pose_by_obb_torch(
                    pose=batch_obj_pose_T,
                    size=size,
                    approaching=self.PICK_GRASP_APPROACHING,
                    target_closing=self.PICK_GRASP_TARGET_CLOSING,
                    depth=self.PICK_GRASP_DEPTH,
                    ortho=self.PICK_GRASP_ORTHO,
                )

            out_pose_vec7[local_pos_tensor] = self._pose_matrices_to_vec7(grasp_T)

        return out_pose_vec7
    
    def get_pick_place_target_pose_world(self):
        """
        返回：
        - target_pose_world: [num_envs, 7]
            - pick: 目标抓取位姿（世界坐标系）
            - place: 放置目标中心 pose（世界坐标系）
        - is_pick: [num_envs] bool

        若当前任一环境的子任务不是 PickSubtask / PlaceSubtask，则抛 NotImplementedError。
        """
        target_pose_world = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float32
        )
        is_pick = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )

        for subtask_num in currently_running_subtasks.tolist():
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            subtask = self.task_plan[subtask_num]

            if isinstance(subtask, PickSubtask):
                target_pose_world[env_idx] = self._get_pick_grasp_pose_world_for_subtask(
                    subtask_num=subtask_num,
                    env_idx=env_idx,
                )
                is_pick[env_idx] = True

            elif isinstance(subtask, PlaceSubtask):
                target_goal = self.subtask_goals[subtask_num]
                if target_goal is None:
                    raise RuntimeError(
                        f"PlaceSubtask at index {subtask_num} has no goal actor"
                    )

                # 防止 place 目标点过低的问题
                place_pose = Pose(target_goal.pose.raw_pose)
                place_pose = place_pose * Pose.create_from_pq(p = torch.tensor([-0.1, 0, 0]))

                target_pose_world[env_idx] = vectorize_pose(place_pose)[env_idx]
                is_pick[env_idx] = False

            else:
                raise NotImplementedError(
                    f"Only PickSubtask and PlaceSubtask are supported, "
                    f"but got {subtask.type} ({type(subtask)}) at subtask {subtask_num}"
                )

        return target_pose_world, is_pick
    
    def get_special_pose(self, pose_name: Optional[str] = None):
        if pose_name == "tcp" or pose_name == None:
            use_pose = self.agent.robot.find_link_by_name(FETCH_TCP_LINK_NAME).pose
        elif pose_name == "target":
            target_pose_world, is_pick = self.get_pick_place_target_pose_world()
            use_pose = Pose(target_pose_world)
        else:
            raise NotImplementedError("尚未实现引用其他位姿")
        
        return use_pose

    ### 通用

    def step(self, action):
        # 缓存用户真正传进来的动作
        if isinstance(action, torch.Tensor):   # torch.Tensor
            self._cur_action[:] = action.clone()
        else:
            self._cur_action[:] = torch.tensor(action, device = self.device)

        step_result = super().step(action)

        if self.policy_info_enable:
            if self.tcp_point is not None:
                self.tcp_point.set_pose(self.agent.robot.find_link_by_name(FETCH_TCP_LINK_NAME).pose)
        # 更新上一时刻动作 (使用拷贝赋值防止泄露)
        self._last_action = self._cur_action.clone()

        return step_result

    def _common_initalize_episode(self, env_idx: torch.Tensor, options):
        with torch.device(self.device):
            with torch.no_grad():
                # 新重置的环境上一步动作为 0
                if getattr(self, "_last_action", None) is None:
                    self._last_action = torch.zeros((self.num_envs, self.agent.single_action_space.shape[0])) # type: ignore
                else:
                    # 假定 step 均为 Tensor
                    self._last_action[env_idx] = 0
                # 新重置的环境当前步动作为 0
                if getattr(self, "_cur_action", None) is None:
                    self._cur_action = torch.zeros((self.num_envs, self.agent.single_action_space.shape[0])) # type: ignore
                else:
                    # 假定 step 均为 Tensor
                    self._cur_action[env_idx] = 0

                # 使 evaluate() 的缓存失效，确保重置后首次调用会重新计算
                self._last_eval_elapsed_steps = None
                self._cached_eval_result = None

    ### Pi0

    # def _pi0_pose_to_state(self, pose: Pose, grasp_qpos: torch.Tensor):
    #     rot = pose.get_q()
    #     if self.pi0_state_rot_type == "axis_angle":
    #         rot, _ = quat_wxyz_tensor_to_axis_angle(rot)
    #     elif self.pi0_state_rot_type == "rot_6d":
    #         rot = rotation_conversions.quaternion_to_matrix(rot)
    #         rot = rotation_conversions.matrix_to_rotation_6d(rot)
    #     elif self.pi0_state_rot_type == "quat":
    #         pass
    #     else:
    #         raise NotImplemented(f"Unknown pi0_state_rot_type: {self.pi0_state_rot_type}")

    #     return torch.concat(
    #         [pose.get_p(), rot, grasp_qpos.unsqueeze(dim = 1)], dim = 1
    #     )

    # def _pi0_trans_to_action(self, trans: Pose, grasp_action: torch.Tensor):
    #     rot = trans.get_q()
    #     if self.pi0_action_rot_type == "axis_angle":
    #         rot, _ = quat_wxyz_tensor_to_axis_angle(rot)
    #     elif self.pi0_action_rot_type == "rot_6d":
    #         rot = rotation_conversions.quaternion_to_matrix(rot)
    #         rot = rotation_conversions.matrix_to_rotation_6d(rot)
    #     elif self.pi0_action_rot_type == "rpy_euler":
    #         rot = quaternion_to_rpy_eular(rot)
    #     else:
    #         raise NotImplemented(f"Unknown pi0_action_rot_type: {self.pi0_action_rot_type}")

    #     return torch.concat(
    #         [trans.get_p(), rot, grasp_action.unsqueeze(dim = 1)], dim = 1
    #     )

    def _pi0_initalize_episode(self, env_idx: torch.Tensor, options):
        # GPU sim 下显式同步
        if self.gpu_sim_enabled:
            # 某些版本文档写的是 gpu_apply_all / gpu_fetch_all，
            # 但官方示例里常见的是带下划线的方法
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

        # 初始化位姿张量
        if not hasattr(self, "last_base_pose_raw"):
            self.last_base_pose_raw = torch.zeros((self.num_envs, 7))
        # 保存上一步世界坐标系下的底盘位姿 (_initialize_episode 自动过滤)
        self.last_base_pose_raw[env_idx] = self.agent.robot.find_link_by_name(FETCH_BASE_LINK_NAME).pose.raw_pose

        # 初始化位姿张量
        if not hasattr(self, "last_tcp_pose_raw"):
            self.last_tcp_pose_raw = torch.zeros((self.num_envs, 7))
        # 保存上一步世界坐标系下的 TCP 位姿 (_initialize_episode 自动过滤)
        self.last_tcp_pose_raw[env_idx] = self.agent.robot.find_link_by_name(FETCH_TCP_LINK_NAME).pose.raw_pose

        # 缓存上一步夹爪状态
        self.gripper_state_tl = torch.mean(self.agent.robot.get_qpos()[:, FETCH_GRIPPER_QPOS_IDX], dim = 1)

    def _pi0_evaluate(self):
        ### 转化为 Pi0 标准动作

        ## 本体观测：移动底盘坐标系下的末端位置、四元数姿态、当前夹爪张开程度（tl: 上一时刻, tc：当前时刻）
        w_b_tl_pose = Pose(self.last_base_pose_raw)
        w_eef_tl_pose = Pose(self.last_tcp_pose_raw)
        b_tl_w_pose = w_b_tl_pose.inv()
        # 上一时刻基座下的末端执行器作为状态
        b_tl_eef_tl_pose = b_tl_w_pose * w_eef_tl_pose
        # 整理为 Pi0 本体观测 (LeRobot 使用的是动作执行前的观测, Evaluate 直接获取的为动作执行后的状态, 因此此处使用 tl)
        # print(f"self.gripper_state_tl: {self.gripper_state_tl.shape}")
        # print(f"b_tl_eef_tl_pose.get_p(): {b_tl_eef_tl_pose.get_p().shape}") 

        # 获取当前底盘与夹爪末端位姿
        w_b_tc_pose = self.agent.robot.find_link_by_name(FETCH_BASE_LINK_NAME).pose
        w_eef_tc_pose = self.agent.robot.find_link_by_name(FETCH_TCP_LINK_NAME).pose   

        # 获取当前夹爪状态 (使用两个夹爪位移的均值, 不进行标准化)
        gripper_state_tc = torch.mean(self.agent.robot.get_qpos()[:, FETCH_GRIPPER_QPOS_IDX], dim = 1)

        # 非推理模式下将收集动作, 使用 s_{t-1} 状态
        if not self.pi0_is_infer_mode:
            #TODO: 夹爪使用实际的夹爪动作
            # pi0_eef_state_tl = torch.concat(
            #     [b_tl_eef_tl_pose.get_p(), b_tl_eef_tl_pose.get_q(), self.gripper_state_tl.unsqueeze(dim = 1)], dim = 1
            # )
            pi0_eef_state_tl = pose_to_target_type(
                b_tl_eef_tl_pose, self.pi0_state_rot_type, [self.gripper_state_tl.unsqueeze(dim = 1)]
            )

            # 动作：移动底盘坐标系下的末端位移变换（位置 + 固定欧拉角）+ 下一时刻夹爪张开程度
            # 获取基于运动坐标系的 Delta 动作
            eef_tl_trans = w_eef_tl_pose.inv() * w_eef_tc_pose
            # 整理为 Pi0 动作
            # pi0_eef_ref_action = torch.concat(
            #     [eef_tl_trans.get_p(), quaternion_to_rpy_eular(eef_tl_trans.get_q()), self._cur_action[:, FETCH_GRIPPER_ACT_IDX].unsqueeze(dim = 1)], dim = 1
            # )
            # pi0_eef_ref_action = self._pi0_trans_to_action(eef_tl_trans, self._cur_action[:, FETCH_GRIPPER_ACT_IDX])
            pi0_eef_ref_action = pose_to_target_type(
                eef_tl_trans, self.pi0_action_rot_type, [self._cur_action[:, FETCH_GRIPPER_ACT_IDX].unsqueeze(dim = 1)]
            )

            # 获取基于基座坐标系的 Delta 动作
            b_tl_eef_tc_pose = b_tl_w_pose * w_eef_tc_pose
            b_tl_trans = b_tl_eef_tc_pose * b_tl_eef_tl_pose.inv()
            # 整理为 Pi0 动作
            # pi0_eef_abs_action = torch.concat(
            #     [b_tl_trans.get_p(), quaternion_to_rpy_eular(b_tl_trans.get_q()), self._cur_action[:,FETCH_GRIPPER_ACT_IDX].unsqueeze(dim = 1)], dim = 1
            # )
            # pi0_eef_abs_action = self._pi0_trans_to_action(b_tl_trans, self._cur_action[:, FETCH_GRIPPER_ACT_IDX])
            pi0_eef_abs_action = pose_to_target_type(
                b_tl_trans, self.pi0_action_rot_type, [self._cur_action[:, FETCH_GRIPPER_ACT_IDX].unsqueeze(dim = 1)]
            )

            res_info = dict(
                pi0_eef_state = pi0_eef_state_tl,
                pi0_eef_ref_action = pi0_eef_ref_action,
                pi0_eef_abs_action = pi0_eef_abs_action,
            )
        else:
        # 推理模式下使用 s_t 状态, 动作赋为 None

            # 整理为当前时刻的 pi0 本体观测
            b_tc_eef_tc_pose = w_b_tc_pose.inv() * w_eef_tc_pose
            # pi0_eef_state_tc = torch.concat(
            #     [b_tc_eef_tc_pose.get_p(), b_tc_eef_tc_pose.get_q(), gripper_state_tc.unsqueeze(dim = 1)], dim = 1
            # )
            pi0_eef_state_tc = pose_to_target_type(
                b_tc_eef_tc_pose, self.pi0_state_rot_type, [gripper_state_tc.unsqueeze(dim = 1)]
            )

            res_info = dict(
                pi0_eef_state = pi0_eef_state_tc,
                # pi0_eef_ref_action = None,
                # pi0_eef_abs_action = None,
            )

        # 更新记录
        self.last_base_pose_raw = w_b_tc_pose.raw_pose
        self.last_tcp_pose_raw = w_eef_tc_pose.raw_pose
        self.gripper_state_tl = gripper_state_tc
        return res_info

    ### policy

    def set_policy_goal_pose(
        self,
        # 传入位姿均相对世界坐标系
        new_pose: Optional[Union[Pose, Literal[
            "tcp", # 当前 TCP 位姿
            "target", # 子任务目标位姿
            "reset" # 任务结束位姿
        ]]] = None,
        env_idx: Optional[int] = None
    ):
        if not isinstance(new_pose, Pose):
            use_pose = self.get_special_pose(new_pose)
        else:
            use_pose = new_pose

        goal_pose = self.goal_point.pose
        if env_idx is not None:
            goal_pose.raw_pose[env_idx] = use_pose.raw_pose[env_idx]
        else:
            goal_pose = use_pose
        
        self.goal_point.set_pose(goal_pose)

    def get_policy_goal_pose(
        self
    ):
        return self.goal_point.pose

    GOAL_POINT_VIS_RADIUS = 0.01
    GOAL_POINT_VIS_LENGTH = 0.05
    GOAL_POINT_VIS_COLOR = (np.array([12, 160, 42], dtype = np.float64) / 255).tolist()
    def _policy_load_scene(self, options):

        # 目标识别体
        goal_point_builder = self.scene.create_actor_builder()
        
        if self.policy_show_goal_axis:
            goal_point_builder = make_vis_axis(goal_point_builder)
        else:
            goal_point_builder.add_cylinder_visual(
                radius = self.GOAL_POINT_VIS_RADIUS,
                half_length = self.GOAL_POINT_VIS_LENGTH,
                material = self.GOAL_POINT_VIS_COLOR,
                pose = sapien.Pose(p = [0, 0, 0])
            )

        goal_point_builder.set_initial_pose(sapien.Pose(p = [0, 0, 0], q = [1, 0, 0, 0]))
        self.goal_point = goal_point_builder.build_kinematic(name = "goal_point")
        self._hidden_objects.append(self.goal_point)

        # 末端指示体
        if self.policy_show_goal_axis:
            tcp_point_builder = self.scene.create_actor_builder()
            tcp_point_builder = make_vis_axis(tcp_point_builder)

            tcp_point_builder.set_initial_pose(sapien.Pose(p = [0, 0, 0], q = [1, 0, 0, 0]))
            self.tcp_point = tcp_point_builder.build_kinematic(name = "tcp_point")
            self._hidden_objects.append(self.tcp_point)

        else:
            self.tcp_point = None

    def _policy_evaluate(self):
        
        ### 相机位姿
        base_world_pose = self.agent.robot.find_link_by_name(FETCH_BASE_LINK_NAME).pose.inv()
        # 头部相机坐标系在基座下的座标系
        world_head_camera_pose = self.agent.robot.find_link_by_name(FETCH_HEAD_CAMERA_LINK).pose
        # 手部相机坐标系在基座下的坐标系
        world_gripper_camera_pose = self.agent.robot.find_link_by_name(FETCH_GRIPPER_CAMERA_LINK).pose

        addition_info = dict(
            # MSHAB 对原始 qpos 进行了裁剪, 此处重新获取原始 qpos 便于与已有模块衔接
            qpos = self.agent.robot.get_qpos(),
            last_action = self._last_action,

            head_camera_t = (base_world_pose * world_head_camera_pose).to_transformation_matrix(),
            gripper_camera_t = (base_world_pose * world_gripper_camera_pose).to_transformation_matrix(),
        )

        if not self.policy_bc_policy_mode:
            ### 角度检查
            eef_pose = self.agent.robot.find_link_by_name(FETCH_TCP_LINK_NAME).pose
            pose_diff = eef_pose.inv() * self.goal_point.pose # type: ignore
            assert isinstance(pose_diff, Pose)

            # 使用底盘坐标系下的差向量而不是末端或目标
            base_pose_inv = self.agent.robot.find_link_by_name(FETCH_BASE_LINK_NAME).pose.inv()
            base_eef_pose = base_pose_inv * eef_pose
            base_goal_pose = base_pose_inv * self.goal_point.pose
            
            loc_diff = base_eef_pose.get_p() - base_goal_pose.get_p()
            loc_error = torch.norm(loc_diff, dim = 1)
            # print(f"loc_diff: {loc_diff}")
            rotvec_diff, rot_error = quat_wxyz_tensor_to_axis_angle(pose_diff.get_q())

            # print(f"rotvec_diff: {rotvec_diff}")
            # print(f"loc_diff: {loc_diff}")

            ### 底盘信息
            # 底盘信息 (统一为底盘坐标系, 不考虑 z 方向, 即 XOY 平面投影)
            base_pose = self.agent.robot.find_link_by_name(FETCH_BASE_LINK_NAME).pose
            base_pose_inv = base_pose.inv()

            if getattr(self, "robot_forward_in_base_link_tensor", None) is None:
                self.robot_forward_in_base_link_tensor = torch.as_tensor(
                    FETCH_FORWARD_IN_BASE_LINK, dtype=torch.float32, device=self.device
                )
            # 认为底盘紧贴底面, z 分量为 (0, 0, 1)
            goal_loc = (base_pose_inv * self.goal_point.pose).get_p()[:, 0:2]
            # (保留以兼容旧的奖励函数) 认为底盘紧贴底面, 底盘前进方向即底盘坐标系 (1, 0, 0) 方向, z 分量为 (0, 0, 1)
            base_forward = torch.zeros_like(goal_loc) # base_T[:, :2, :2] @ self.robot_forward_in_base_link_tensor
            base_forward[:, 0] = 1
            # (保留以兼容旧的奖励函数) 使用底盘坐标系时 base_loc 总为 (0, 0)
            base_loc = torch.zeros_like(goal_loc)# base_T[:, 0:2, 3]

            base2goal_vec = torch.as_tensor(goal_loc - base_loc)

            addition_info.update(dict(
                rot_error = rot_error,
                loc_error = loc_error,
                
                loc_diff = loc_diff,
                rotvec_diff = rotvec_diff,

                base_forward = base_forward,
                base2goal_vec = base2goal_vec,
            ))

        return addition_info
    
    ### 载入容器信息
    TARGET_RECEPTACLE_MAX_BYTES = 64

    def _get_current_target_receptacles_info(self):
        current_target_receptacles = ["" for _ in range(self.num_envs)]

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )

        for subtask_num in currently_running_subtasks.tolist():
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            subtask = self.task_plan[subtask_num]

            merged_target_receptacles = getattr(subtask, "target_receptacles", None)
            if merged_target_receptacles is None:
                continue

            assert len(merged_target_receptacles) == self.num_envs

            for env_id in env_idx.tolist():
                receptacle = merged_target_receptacles[env_id]
                current_target_receptacles[env_id] = receptacle or ""

        encoded = [s.encode("utf-8") for s in current_target_receptacles]

        # 防止固定长度字符串被静默截断
        max_len = max(len(x) for x in encoded) if encoded else 0
        assert max_len <= self.TARGET_RECEPTACLE_MAX_BYTES, (
            f"target_receptacle length {max_len} exceeds "
            f"TARGET_RECEPTACLE_MAX_BYTES={self.TARGET_RECEPTACLE_MAX_BYTES}"
        )

        return np.asarray(encoded, dtype=f"S{self.TARGET_RECEPTACLE_MAX_BYTES}")

    ### 位姿进行可视化函数

    def _vis_pose_load_scene(self, options):
        if self.num_vis_pose <= 0:
            return

        self.vis_pose_actor_list: list[Actor] = []
        for i in range(self.num_vis_pose):
            vis_pose_builder = self.scene.create_actor_builder()
            vis_pose_builder = make_vis_axis(vis_pose_builder)

            vis_pose_builder.set_initial_pose(sapien.Pose(p = [0, 0, 0], q = [1, 0, 0, 0]))
            vis_pose_actor = vis_pose_builder.build_kinematic(name = f"vis_pose_{i}")

            self.vis_pose_actor_list.append(vis_pose_actor)
            self._hidden_objects.append(vis_pose_actor)

    def reset_vis_pose(
        self
    ):
        if self.num_vis_pose <= 0:
            warn(f"vis pose is disable as num_vis_pose is {self.num_vis_pose}")
            return
        
        for i in range(self.num_vis_pose):
            self.vis_pose_actor_list[i].set_pose(sapien.Pose())

    def set_vis_pose(
        self,
        vis_pose: Sequence[Union[sapien.Pose, Pose]]
    ):
        '''
        对位姿进行可视化, 仅可视化前 self.num_vis_pose 个位姿
        '''
        if self.num_vis_pose <= 0:
            warn(f"vis pose is disable as num_vis_pose is {self.num_vis_pose}")
            return
        
        num_use_pose = min(len(vis_pose), self.num_vis_pose)
        for i in range(num_use_pose):
            self.vis_pose_actor_list[i].set_pose(vis_pose[i])

    ###

    SUPPORTED_ROBOTS = ["fetch", "fetch_modified", "fetch_origin_like"]
    agent: Fetch

    EE_REST_POS_WRT_BASE = (0.5, 0, 1.25)
    pick_cfg = PickSubtaskConfig(
        horizon=200,
        ee_rest_thresh= 0.1, # 0.05,
    )
    place_cfg = PlaceSubtaskConfig(
        horizon=200,
        obj_goal_thresh=0.15,
        ee_rest_thresh= 0.1, # 0.05,
    )
    navigate_cfg = NavigateSubtaskConfig(
        horizon=500,
        ee_rest_thresh=0.05,
        navigated_successfully_dist=2,
    )
    open_cfg = OpenSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        joint_qpos_open_thresh_frac=dict(
            default=0.9,
            fridge=0.75,
            kitchen_counter=0.9,
        ),
    )
    close_cfg = CloseSubtaskConfig(
        horizon=200,
        ee_rest_thresh=0.05,
        joint_qpos_close_thresh_frac=0.01,
    )

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=50,
            gpu_memory_config=GPUMemoryConfig(
                temp_buffer_capacity=2**24,
                max_rigid_contact_count=2**23,
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**21,
            ),
        )

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        task_plans: List[TaskPlan] = [],
        require_build_configs_repeated_equally_across_envs=True,
        randomize_build_configs_per_env=False,
        add_event_tracker_info=False,
        invisible_goals_in_human_render=False,
        task_cfgs=dict(),

        # 启用 pi0 数据集所需的 info
        pi0_info_enable: bool = True,
        # 将 pi0 信息合并到 extra obs 中
        pi0_info_merge_to_extra_obs: bool = False,
        # 推理模式下, 不分析动作, 且输出状态为动作执行后的状态
        pi0_is_infer_mode: bool = False,
        # pi0 观测的姿态表示, 默认为 Rot6D (更适合表示全局姿态), 否则使用轴角对 (与 Libero 保持一致, 更适合表示局部姿态)
        pi0_state_rot_type: ROT_TYPE = "rot_6d",
        # pi0 动作的姿态表示, 默认为轴角对 (与 Libero 保持一致, 更适合表示局部姿态), 否则使用 Rot6D (更适合表示全局姿态)
        pi0_action_rot_type: ROT_TYPE = "axis_angle",

        # 启用 RL Policy 所需的 info
        policy_info_enable: bool = True,
        # 仅记录训练有监督策略所需的信息 (当前仅记录相机变换、last action 与 qpos) 
        policy_bc_policy_mode: bool = False,
        # 将 RL policy 信息合并到 extra obs 中
        policy_info_merge_to_extra_obs: bool = True,
        # 显示末端与目标的坐标系
        policy_show_goal_axis: bool = True,

        # 是否使用详细的成功判断 info (ee_rest 距离与 grasp_angle)
        detailed_success_checker_enable: bool = True,

        # 记录容器信息
        receptacles_enable: bool = True,
        # 最大可视位姿数
        num_vis_pose: int = 0,

        **kwargs,
    ):
        
        self.pi0_info_enable = pi0_info_enable
        self.pi0_is_infer_mode = pi0_is_infer_mode
        self.pi0_info_merge_to_extra_obs = pi0_info_merge_to_extra_obs
        self.pi0_state_rot_type: ROT_TYPE = pi0_state_rot_type
        self.pi0_action_rot_type: ROT_TYPE = pi0_action_rot_type
        if (not self.pi0_info_enable) and self.pi0_info_merge_to_extra_obs:
            warn("将 pi0 信息合并到 extra obs 前需要启用 pi0_info_enable")
            self.pi0_info_merge_to_extra_obs = False
        
        self.policy_info_enable = policy_info_enable
        self.policy_bc_policy_mode = policy_bc_policy_mode
        self.policy_show_goal_axis = policy_show_goal_axis
        self.policy_info_merge_to_extra_obs = policy_info_merge_to_extra_obs
        if (not self.policy_info_enable) and self.policy_info_merge_to_extra_obs:
            warn("将 policy 信息合并到 extra obs 前需要启用 policy_info_enable")
            self.policy_info_merge_to_extra_obs = False
        
        self.detailed_success_checker_enable = detailed_success_checker_enable

        self.receptacles_enable = receptacles_enable
        self.num_vis_pose = num_vis_pose

        self.task_cfgs: Dict[str, SubtaskConfig] = dict(
            pick=self.pick_cfg,
            place=self.place_cfg,
            navigate=self.navigate_cfg,
            open=self.open_cfg,
            close=self.close_cfg,
        )

        task_cfg_update_dict = task_cfgs
        for k, v in task_cfg_update_dict.items():
            self.task_cfgs[k].update(v)

        assert all_equal(
            [len(plan.subtasks) for plan in task_plans]
        ), "All parallel task plans must be the same length"
        assert all(
            [
                all_same_type(parallel_subtasks)
                for parallel_subtasks in zip(*[plan.subtasks for plan in task_plans])
            ]
        ), "All parallel task plans must have same subtask types in same order"

        if randomize_build_configs_per_env:
            assert (
                not require_build_configs_repeated_equally_across_envs
            ), f"Received {randomize_build_configs_per_env=} but cannot randomize build configs per env with {require_build_configs_repeated_equally_across_envs=}"
        self._require_build_configs_repeated_equally_across_envs = (
            require_build_configs_repeated_equally_across_envs
        )
        self._randomize_build_configs_per_env = randomize_build_configs_per_env
        self._add_event_tracker_info = add_event_tracker_info
        self._invisible_goals_in_human_render = invisible_goals_in_human_render

        self.base_task_plans = dict(
            (tuple([subtask.uid for subtask in tp.subtasks]), tp) for tp in task_plans
        )
        self.bc_to_task_plans: Dict[str, List[TaskPlan]] = defaultdict(list)
        for tp in task_plans:
            self.bc_to_task_plans[tp.build_config_name].append(tp)

        self._init_config_names = set([tp.init_config_name for tp in task_plans])

        self.tp0 = task_plans[0]

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # -------------------------------------------------------------------------------------------------
    # PROCESS TASKS
    # -------------------------------------------------------------------------------------------------
    
    def _merge_target_receptacles(
        self,
        parallel_subtasks: List[Subtask],
    ) -> Optional[List[str]]:
        """
        单环境子任务里 target_receptacles 通常长度为 1；
        merge 后返回长度为 num_envs 的 list[str]。
        旧 task plan 或未提供该字段时返回 None。
        """
        merged = []
        has_any = False

        for subtask in parallel_subtasks:
            tr = getattr(subtask, "target_receptacles", None)
            if tr is None or len(tr) == 0:
                merged.append("")
            else:
                assert len(tr) == 1, (
                    "Expected single-env subtask.target_receptacles to have length 1, "
                    f"but got {tr}"
                )
                merged.append(tr[0])
                has_any = True

        return merged if has_any else None

    def _merge_pick_subtasks(
        self, subtask_num: int, parallel_subtasks: List[PickSubtask]
    ):
        merged_obj_name = f"obj_{subtask_num}"
        self.subtask_objs.append(
            self._create_merged_actor_from_subtasks(
                parallel_subtasks, name=merged_obj_name,
            )
        )
        self.subtask_goals.append(None)

        # NOTE (arth): this is a bit tricky, since prepare_groceries sometimes has an articulation config
        #   (pick from fridge), but sometimes does not (pick from countertop). however, when running tasks
        #   sequentially, the prepare_groceries parallel_subtasks will either all have or all not have
        #   articulation_config. this is unlike tasks like tidy_house, which never has articulation_config
        #   or set_table which always has articulation_config.
        #   later, we expect that all subtask_x has len(_objs) == num_envs, so we can't merge for prepare_groceries
        #   as-is. for now, we ignore the prepare_groceries case (since fridge is opened by ao_config anyways)
        #   but in the future we'll need to support a case with e.g. a modified set_table with pick from table
        if all(
            [subtask.articulation_config is not None for subtask in parallel_subtasks]
        ):
            merged_articulation_name = f"articulation-{subtask_num}"
            merged_articulation = (
                self._create_merged_articulation_from_articulation_ids(
                    [
                        subtask.articulation_config.articulation_id
                        for subtask in parallel_subtasks
                    ],
                    name=merged_articulation_name,
                    merging_different_articulations=True,
                )
            )
            self.subtask_articulations.append(merged_articulation)
        else:
            self.subtask_articulations.append(None)

        self.task_plan.append(
            PickSubtask(
                obj_id=merged_obj_name,
                # NOTE (arth): pick subtask might have different kinds of articulations
                #       merged into one Articulation view (e.g. fridge vs kitchen_counter)
                #       in this case, ArticulationConfig attributes like handle_link_idx
                #       don't make much sense (and aren't needed by the pick task anyways)
                articulation_config=None,
                source_obj_ids=[subtask.obj_id for subtask in parallel_subtasks],
                target_receptacles=self._merge_target_receptacles(parallel_subtasks),
            )
        )

    def _merge_place_subtasks(
        self,
        env_idx: torch.Tensor,
        subtask_num: int,
        parallel_subtasks: List[PlaceSubtask],
    ):
        merged_obj_name = f"obj_{subtask_num}"
        self.subtask_objs.append(
            self._create_merged_actor_from_subtasks(
                parallel_subtasks, name=merged_obj_name
            )
        )
        self.subtask_goals.append(self.premade_goal_list[subtask_num])

        merged_goal_pos = common.to_tensor(
            [subtask.goal_pos for subtask in parallel_subtasks]
        )
        merged_goal_rectangle_corners = common.to_tensor(
            [subtask.goal_rectangle_corners for subtask in parallel_subtasks]
        )

        self.subtask_goals[-1].set_pose(
            Pose.create_from_pq(q=GOAL_POSE_Q, p=merged_goal_pos[env_idx])
        )

        # NOTE (arth): see notes above regarding merged articulation config
        if all(
            [subtask.articulation_config is not None for subtask in parallel_subtasks]
        ):
            merged_articulation_name = f"articulation-{subtask_num}"
            merged_articulation = (
                self._create_merged_articulation_from_articulation_ids(
                    [
                        subtask.articulation_config.articulation_id
                        for subtask in parallel_subtasks
                    ],
                    name=merged_articulation_name,
                    merging_different_articulations=True,
                )
            )
            self.subtask_articulations.append(merged_articulation)
        else:
            self.subtask_articulations.append(None)

        self.task_plan.append(
            PlaceSubtask(
                obj_id=merged_obj_name,
                goal_pos=merged_goal_pos,
                goal_rectangle_corners=merged_goal_rectangle_corners,
                validate_goal_rectangle_corners=False,
                articulation_config=None,
                target_receptacles=self._merge_target_receptacles(parallel_subtasks),
            )
        )
        self.check_progressive_success_subtask_nums.append(subtask_num)

    def _merge_navigate_subtasks(
        self,
        env_idx: torch.Tensor,
        last_subtask0: Subtask,
        subtask_num: int,
        parallel_subtasks: List[NavigateSubtask],
    ):
        self.subtask_goals.append(None)
        self.subtask_articulations.append(None)

        if isinstance(last_subtask0, PickSubtask):
            last_subtask_obj = self.subtask_objs[-1]
            self.subtask_objs.append(last_subtask_obj)
            self.task_plan.append(
                NavigateSubtask(
                    obj_id=last_subtask_obj.name,
                )
            )
        else:
            self.subtask_objs.append(None)
            self.task_plan.append(NavigateSubtask())

    def _merge_open_subtasks(
        self, subtask_num: int, parallel_subtasks: List[OpenSubtask]
    ):
        subtask0 = parallel_subtasks[0]

        # NOTE (arth): current MS3 requires all parallel articulations be the same
        assert all_equal([subtask.articulation_type for subtask in parallel_subtasks])
        assert all_equal(
            [subtask.articulation_handle_link_idx for subtask in parallel_subtasks]
        )
        assert all_equal(
            [
                subtask.articulation_handle_active_joint_idx
                for subtask in parallel_subtasks
            ]
        )

        merged_obj_name = f"obj_{subtask_num}"
        self.subtask_objs.append(
            self._create_merged_actor_from_subtasks(
                parallel_subtasks, name=merged_obj_name
            )
        )
        self.subtask_goals.append(self.premade_goal_list[subtask_num])

        merged_articulation_name = f"articulation-{subtask_num}"
        merged_articulation_relative_handle_pose = Pose.create_from_pq(
            p=[
                subtask.articulation_relative_handle_pos
                for subtask in parallel_subtasks
            ]
        )
        merged_articulation = self._create_merged_articulation_from_subtasks(
            parallel_subtasks, name=merged_articulation_name
        )
        self.subtask_articulations.append(merged_articulation)

        self.task_plan.append(
            OpenSubtask(
                obj_id=merged_obj_name,
                articulation_type=subtask0.articulation_type,
                articulation_id=merged_articulation_name,
                articulation_handle_link_idx=subtask0.articulation_handle_link_idx,
                articulation_handle_active_joint_idx=subtask0.articulation_handle_active_joint_idx,
                articulation_relative_handle_pos=merged_articulation_relative_handle_pose,
            )
        )

    def _merge_close_subtasks(
        self, subtask_num: int, parallel_subtasks: List[CloseSubtask]
    ):
        subtask0 = parallel_subtasks[0]

        # NOTE (arth): current MS3 requires all parallel articulations be the same
        assert all_equal([subtask.articulation_type for subtask in parallel_subtasks])
        assert all_equal(
            [subtask.articulation_handle_link_idx for subtask in parallel_subtasks]
        )
        assert all_equal(
            [
                subtask.articulation_handle_active_joint_idx
                for subtask in parallel_subtasks
            ]
        )

        self.subtask_objs.append(None)
        self.subtask_goals.append(self.premade_goal_list[subtask_num])

        merged_articulation_name = f"articulation-{subtask_num}"
        merged_articulation_relative_handle_pose = Pose.create_from_pq(
            p=[
                subtask.articulation_relative_handle_pos
                for subtask in parallel_subtasks
            ]
        )
        merged_articulation = self._create_merged_articulation_from_subtasks(
            parallel_subtasks, name=merged_articulation_name
        )
        self.subtask_articulations.append(merged_articulation)

        # NOTE (arth): currently a band-aid solution, teleport bowl inside kitchen drawer away
        merged_removed_obj_name = None
        if subtask0.remove_obj_id is not None:
            merged_removed_obj_name = f"remove_obj_{subtask_num}"
            self._create_merged_actor_from_obj_ids(
                [subtask.remove_obj_id for subtask in parallel_subtasks],
                name=f"remove_obj_{subtask_num}",
            ).set_pose(
                Pose.create_from_pq(p=[-10_000, -10_000, -9000])
            )  # assume one entity per scene

        self.task_plan.append(
            CloseSubtask(
                articulation_type=subtask0.articulation_type,
                articulation_id=merged_articulation_name,
                articulation_handle_link_idx=subtask0.articulation_handle_link_idx,
                articulation_handle_active_joint_idx=subtask0.articulation_handle_active_joint_idx,
                articulation_relative_handle_pos=merged_articulation_relative_handle_pose,
                remove_obj_id=merged_removed_obj_name,
            )
        )

    def process_task_plan(
        self,
        env_idx: torch.Tensor,
        sampled_subtask_lists: List[List[Subtask]],
    ):

        self.subtask_objs: List[Actor] = []
        self.subtask_goals: List[Actor] = []
        self.subtask_articulations: List[Articulation] = []
        self.check_progressive_success_subtask_nums: List[int] = []

        # build new merged task_plan and merge actors of parallel task plants
        self.task_plan: List[Subtask] = []
        last_subtask0 = None
        for subtask_num, parallel_subtasks in enumerate(zip(*sampled_subtask_lists)):
            composite_subtask_uids = [subtask.uid for subtask in parallel_subtasks]
            subtask0: Subtask = parallel_subtasks[0]

            if isinstance(subtask0, PickSubtask):
                self._merge_pick_subtasks(subtask_num, parallel_subtasks)
            elif isinstance(subtask0, PlaceSubtask):
                self._merge_place_subtasks(env_idx, subtask_num, parallel_subtasks)
            elif isinstance(subtask0, NavigateSubtask):
                self._merge_navigate_subtasks(
                    env_idx, last_subtask0, subtask_num, parallel_subtasks
                )
            elif isinstance(subtask0, OpenSubtask):
                self._merge_open_subtasks(subtask_num, parallel_subtasks)
            elif isinstance(subtask0, CloseSubtask):
                self._merge_close_subtasks(subtask_num, parallel_subtasks)
            else:
                raise AttributeError(
                    f"{subtask0.type} {type(subtask0)} not yet supported"
                )

            last_subtask0 = subtask0

            self.task_plan[-1].composite_subtask_uids = composite_subtask_uids

        # add navigation goals for each Navigate Subtask depending on following subtask
        last_subtask = None
        for i, (subtask_obj, subtask_goal, subtask_articulation, subtask) in enumerate(
            zip(
                self.subtask_objs,
                self.subtask_goals,
                self.subtask_articulations,
                self.task_plan,
            )
        ):
            if isinstance(last_subtask, NavigateSubtask):
                if isinstance(subtask, PickSubtask):
                    self.subtask_goals[i - 1] = subtask_obj
                elif isinstance(subtask, PlaceSubtask):
                    self.subtask_goals[i - 1] = subtask_goal
                elif isinstance(subtask, OpenSubtask) or isinstance(
                    subtask, CloseSubtask
                ):
                    self.subtask_goals[i - 1] = (
                        subtask_articulation.links[subtask.articulation_handle_link_idx]
                        if subtask.articulation_type == "kitchen_counter"
                        else subtask_articulation
                    )
                    last_subtask.articulation_config = ArticulationConfig(
                        articulation_id=subtask.articulation_id,
                        articulation_type=subtask.articulation_type,
                        articulation_handle_link_idx=subtask.articulation_handle_link_idx,
                        articulation_handle_active_joint_idx=subtask.articulation_handle_active_joint_idx,
                    )
            last_subtask = subtask

        assert len(self.subtask_objs) == len(self.task_plan)
        assert len(self.subtask_goals) == len(self.task_plan)
        assert len(self.subtask_articulations) == len(self.task_plan)

        self.task_horizons = torch.tensor(
            [self.task_cfgs[subtask.type].horizon for subtask in self.task_plan],
            device=self.device,
            dtype=torch.long,
        )
        self.task_ids = torch.tensor(
            [self.task_cfgs[subtask.type].task_id for subtask in self.task_plan],
            device=self.device,
            dtype=torch.long,
        )

    def _get_actor_entity(self, actor_id: str, env_num: int):
        actor = self.scene_builder.movable_objects[actor_id]
        return actor._objs[actor._scene_idxs.tolist().index(env_num)]

    def _create_merged_actor_from_obj_ids(
        self,
        obj_ids: List[str],
        name: str = None,
    ):
        merged_obj = Actor.create_from_entities(
            [
                self._get_actor_entity(actor_id=f"env-{i}_{oid}", env_num=i)
                for i, oid in enumerate(obj_ids)
            ],
            scene=self.scene,
            scene_idxs=torch.arange(self.num_envs, dtype=int),
        )
        if name is not None:
            merged_obj.name = name
        return merged_obj

    def _create_merged_actor_from_subtasks(
        self,
        parallel_subtasks: List[Union[PickSubtask, PlaceSubtask]],
        name: str = None,
    ):
        return self._create_merged_actor_from_obj_ids(
            [subtask.obj_id for subtask in parallel_subtasks], name
        )

    def _get_articulation_entity(self, articulation_id: str, env_num: int):
        ms_articulation = self.scene_builder.articulations[articulation_id]
        return ms_articulation._objs[
            ms_articulation._scene_idxs.tolist().index(env_num)
        ]

    def _create_merged_articulation_from_articulation_ids(
        self,
        articulation_ids: List[str],
        name: str = None,
        merging_different_articulations: bool = False,
    ):
        scene_idx_to_physx_articulation_objs = [None for _ in range(self.num_envs)]
        for env_num, aid in enumerate(articulation_ids):
            scene_idx_to_physx_articulation_objs[env_num] = (
                self._get_articulation_entity(f"env-{env_num}_{aid}", env_num)
            )
        merged_articulation = Articulation.create_from_physx_articulations(
            scene_idx_to_physx_articulation_objs,
            scene=self.scene,
            scene_idxs=torch.arange(self.num_envs),
            _merged=merging_different_articulations,
        )
        merged_articulation.name = name
        return merged_articulation

    def _create_merged_articulation_from_subtasks(
        self,
        parallel_subtasks: List[Union[OpenSubtask, CloseSubtask]],
        name: str = None,
        merging_different_articulations: bool = False,
    ):
        return self._create_merged_articulation_from_articulation_ids(
            [subtask.articulation_id for subtask in parallel_subtasks],
            name,
            merging_different_articulations,
        )

    def _make_goal(
        self,
        pos: Union[Tuple[float, float, float], List[Tuple[float, float, float]]] = None,
        radius=0.15,
        name="goal_site",
        goal_type="sphere",
        color=[0, 1, 0, 1],
    ):
        if pos is not None:
            if len(pos) == self.num_envs:
                initial_pose = Pose.create_from_pq(p=pos)
            else:
                initial_pose = sapien.Pose(p=pos)
        else:
            initial_pose = sapien.Pose()

        if self._invisible_goals_in_human_render:
            color[-1] = 0

        if goal_type == "sphere":
            goal = actors.build_sphere(
                self.scene,
                radius=radius,
                color=color,
                name=name,
                body_type="kinematic",
                add_collision=False,
                initial_pose=initial_pose,
            )
        elif goal_type == "cube":
            goal = actors.build_cube(
                self.scene,
                half_size=radius,
                color=color,
                name=name,
                body_type="kinematic",
                add_collision=False,
                initial_pose=initial_pose,
            )
        elif goal_type == "cylinder":
            goal = actors.build_cylinder(
                self.scene,
                radius=radius,
                half_length=radius,
                color=color,
                name=name,
                body_type="kinematic",
                add_collision=False,
                initial_pose=initial_pose,
            )
        self._hidden_objects.append(goal)
        return goal

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # RESET/RECONFIGURE HANDLING
    # -------------------------------------------------------------------------------------------------

    def _after_reconfigure(self, options):
        force_rew_ignore_links = [
            self.agent.finger1_link,
            self.agent.finger2_link,
        ]
        self.force_articulation_link_ids = [
            link.name
            for link in self.agent.robot.get_links()
            if link not in force_rew_ignore_links
        ]
        self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
        return super()._after_reconfigure(options)

    def _load_scene(self, options):
        self.premade_goal_list: List[Actor] = []
        for subtask_num, subtask in enumerate(self.tp0.subtasks):
            if isinstance(subtask, PlaceSubtask):
                goal = self._make_goal(
                    radius=self.place_cfg.obj_goal_thresh,
                    name=f"goal_{subtask_num}",
                    goal_type=(
                        "cylinder"
                        if self.place_cfg.goal_type == "zone"
                        else self.place_cfg.goal_type
                    ),
                )
            elif isinstance(subtask, OpenSubtask) or isinstance(subtask, CloseSubtask):
                goal = self._make_goal(
                    radius=0.05,
                    name=f"goal_{subtask_num}",
                    goal_type="sphere",
                )
            else:
                goal = None
            self.premade_goal_list.append(goal)

        self.build_config_idx_to_task_plans: Dict[int, List[TaskPlan]] = dict()
        for bc in self.bc_to_task_plans.keys():
            self.build_config_idx_to_task_plans[
                self.scene_builder.build_config_names_to_idxs[bc]
            ] = self.bc_to_task_plans[bc]

        num_bcis = len(self.build_config_idx_to_task_plans.keys())

        assert (
            not self._require_build_configs_repeated_equally_across_envs
            or self.num_envs % num_bcis == 0
        ), f"These task plans cover {num_bcis} build configs, but received {self.num_envs} envs. Either change the task plan list, change num_envs, or set require_build_configs_repeated_equally_across_envs=False. Note if require_build_configs_repeated_equally_across_envs=False and num_envs % num_build_configs != 0, then a) if num_envs > num_build_configs, then some build configs might be built in more parallel envs than others (meaning associated task plans will be sampled more frequently), and b) if num_envs < num_build_configs, then some build configs might not be built at all (meaning associated task plans will not be used)."

        # if num_bcis < self.num_envs, repeat bcis and truncate at self.num_envs
        self.build_config_idxs: List[int] = options.get(
            "build_config_idxs",
            (
                self._episode_rng.choice(
                    list(self.build_config_idx_to_task_plans.keys()),
                    size=self.num_envs,
                    replace=True,
                ).tolist()
                if self._randomize_build_configs_per_env
                else np.repeat(
                    sorted(list(self.build_config_idx_to_task_plans.keys())),
                    np.ceil(self.num_envs / num_bcis),
                )[: self.num_envs].tolist()
            ),
        )
        self.num_task_plans_per_bci = torch.tensor(
            [
                len(self.build_config_idx_to_task_plans[bci])
                for bci in self.build_config_idxs
            ],
            device=self.device,
        )
        self.scene_builder.build(self.build_config_idxs, self._init_config_names)
        self.ee_rest_pos_wrt_base = Pose.create_from_pq(
            p=self.EE_REST_POS_WRT_BASE, device=self.device
        )
        self.subtask_pointer = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.subtask_steps_left = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.ee_rest_goal = self._make_goal(
            radius=0.05,
            name="ee_rest_goal",
            goal_type="sphere",
        )
        ### 用于 RL 策略
        if self.policy_info_enable:
            self._policy_load_scene(options)
        ### 用于 目标定可视化
        if self.num_vis_pose > 0:
            self._vis_pose_load_scene(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options):
        with torch.device(self.device):
            self.robot_cumulative_force[env_idx] = 0

            if env_idx.numel() == self.num_envs:
                self.task_plan_idxs: torch.Tensor = options.get("task_plan_idxs", None)
            if self.task_plan_idxs is None or env_idx.numel() < self.num_envs:
                if self.task_plan_idxs is None:
                    self.task_plan_idxs = torch.zeros(self.num_envs, dtype=torch.int)
                low = torch.zeros(env_idx.numel(), dtype=torch.int)
                high = self.num_task_plans_per_bci[env_idx]
                size = (env_idx.numel(),)
                self.task_plan_idxs[env_idx] = (
                    torch.randint(2**63 - 1, size=size) % (high - low).int() + low.int()
                ).int()
            else:
                self.task_plan_idxs = self.task_plan_idxs.int()
            sampled_task_plans = [
                self.build_config_idx_to_task_plans[bci][tpi]
                for bci, tpi in zip(self.build_config_idxs, self.task_plan_idxs)
            ]
            self.init_config_idxs = [
                self.scene_builder.init_config_names_to_idxs[tp.init_config_name]
                for tp in sampled_task_plans
            ]
            super()._initialize_episode(env_idx, options)
            self.process_task_plan(
                env_idx,
                sampled_subtask_lists=[tp.subtasks for tp in sampled_task_plans],
            )

            self.subtask_pointer[env_idx] = 0
            self.subtask_steps_left[env_idx] = self.task_cfgs[
                self.task_plan[0].type
            ].horizon

            self.resting_qpos = torch.tensor(self.agent.keyframes["rest"].qpos[3:-2])

            # 用于清空上一时刻动作
            self._common_initalize_episode(env_idx, options)
            ### 用于转化 Pi0 标准动作
            if self.pi0_info_enable:
                self._pi0_initalize_episode(env_idx, options)

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # STATE RESET
    # -------------------------------------------------------------------------------------------------

    def get_state_dict(self):
        state_dict = super().get_state_dict()

        state_dict["task_plan_idxs"] = self.task_plan_idxs.clone()
        state_dict["build_config_idxs"] = copy.deepcopy(self.build_config_idxs)
        state_dict["init_config_idxs"] = copy.deepcopy(self.init_config_idxs)

        state_dict["subtask_pointer"] = self.subtask_pointer.clone()
        state_dict["subtask_steps_left"] = self.subtask_steps_left.clone()
        state_dict["robot_cumulative_force"] = self.robot_cumulative_force.clone()

        return state_dict

    def set_state_dict(self, state_dict: Dict):
        task_plan_idxs = common.to_tensor(state_dict.get("task_plan_idxs"))
        build_config_idxs = state_dict.get("build_config_idxs")
        init_config_idxs = state_dict.get("init_config_idxs")

        assert torch.all(
            torch.tensor(self.build_config_idxs) == torch.tensor(build_config_idxs)
        ), f"Please pass the same task plan list when creating this env as was used in this state dict; currently built {self.build_config_idxs=}, state dict {build_config_idxs=}"

        self._initialize_episode(
            torch.arange(self.num_envs), options=dict(task_plan_idxs=task_plan_idxs)
        )

        assert torch.all(
            torch.tensor(self.init_config_idxs) == torch.tensor(init_config_idxs)
        ), f"Please pass the same task plan list when creating this env as was used in this state dict; currently init'd {self.init_config_idxs=}, state dict {init_config_idxs=}"

        self.subtask_pointer = state_dict.get("subtask_pointer")
        self.subtask_steps_left = state_dict.get("subtask_steps_left")
        self.robot_cumulative_force = state_dict.get("robot_cumulative_force")

        super().set_state_dict(state_dict)

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # SUBTASK STATUS CHECKERS/UPDATERS
    # -------------------------------------------------------------------------------------------------

    def evaluate(self):
        # 幂等性保护：同一时间步多次调用时，直接返回缓存结果，
        # 避免 robot_cumulative_force、subtask_steps_left 等状态被重复修改
        cur_elapsed = getattr(self, "_elapsed_steps", None)
        if cur_elapsed is not None:
            last_eval = getattr(self, "_last_eval_elapsed_steps", None)
            cached = getattr(self, "_cached_eval_result", None)
            if last_eval is not None and cached is not None and torch.equal(cur_elapsed, last_eval):
                return cached

        robot_force = (
            self.agent.robot.get_net_contact_forces(self.force_articulation_link_ids)
            .norm(dim=-1)
            .sum(dim=-1)
        )
        self.robot_cumulative_force += robot_force

        # NOTE (arth): update ee_rest_world_pose every step since robot moves
        self.ee_rest_world_pose: Pose = (
            self.agent.base_link.pose * self.ee_rest_pos_wrt_base
        )
        self.handle_world_poses: List[Union[Pose, None]] = []
        for subtask, articulation in zip(self.task_plan, self.subtask_articulations):
            if isinstance(subtask, OpenSubtask) or isinstance(subtask, CloseSubtask):
                self.handle_world_poses.append(
                    articulation.links[subtask.articulation_handle_link_idx].pose
                    * subtask.articulation_relative_handle_pos
                )
            else:
                self.handle_world_poses.append(None)

        subtask_success, success_checkers = self._subtask_check_success()
        progressive_task_success, progressive_task_checkers = (
            self._progressive_task_check_success()
        )

        success_checkers[
            "cumulative_force_within_limit"
        ] |= self.subtask_pointer >= len(self.task_plan)

        move_to_next_subtask = (
            subtask_success
            & success_checkers["cumulative_force_within_limit"]
            & progressive_task_success
        )
        self.subtask_pointer[move_to_next_subtask] += 1
        success = (
            self.subtask_pointer >= len(self.task_plan)
        ) & progressive_task_success
        # set robot_cumulative_force to 0 if evaluating new subtask
        self.robot_cumulative_force[
            move_to_next_subtask & (self.subtask_pointer < len(self.task_plan))
        ] = 0

        self.subtask_steps_left -= 1
        update_subtask_horizon = subtask_success & progressive_task_success & ~success
        self.subtask_steps_left[update_subtask_horizon] = self.task_horizons[
            self.subtask_pointer[update_subtask_horizon]
        ]

        fail = (
            ((self.subtask_steps_left <= 0) & ~success)
            | (~success_checkers["cumulative_force_within_limit"])
            | ~progressive_task_success
        )

        subtask_type = torch.full_like(
            self.subtask_pointer, UNIQUE_SUCCESS_SUBTASK_TYPE
        )
        subtask_type[~success] = self.task_ids[self.subtask_pointer[~success]]

        origin_info = dict(
            success=success,
            fail=fail,
            subtask=self.subtask_pointer,
            subtask_type=subtask_type,
            subtasks_steps_left=self.subtask_steps_left,
            robot_force=robot_force,
            robot_cumulative_force=self.robot_cumulative_force,
            **success_checkers,
            **progressive_task_checkers,
        )

        ### 转化为 Pi0 标准动作
        if self.pi0_info_enable:
            pi0_info = self._pi0_evaluate()
            origin_info.update(pi0_info)

        if self.policy_info_enable:
            policy_info = self._policy_evaluate()
            origin_info.update(policy_info)

        if self.receptacles_enable:
            target_receptacles = self._get_current_target_receptacles_info()
            origin_info.update(target_receptacles = target_receptacles)

        # # Debug
        # target_pose_world, is_pick = self.get_pick_place_target_pose_world()
        # print(f"target_pose_world {target_pose_world}, is_pick {is_pick}")

        # 缓存本次计算结果，供同一时间步的后续调用复用
        self._last_eval_elapsed_steps = cur_elapsed.clone() if cur_elapsed is not None else None
        self._cached_eval_result = origin_info

        return origin_info

    def _progressive_task_check_success(self):
        # check if prior tasks are still in completion state
        # NOTE (arth): while Habitat also checked things like holding vs not,
        #       in this setting only really place requires persisting checkers
        progressive_task_success = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        progressive_task_checkers = dict()
        for subtask_num in self.check_progressive_success_subtask_nums:
            subtask = self.task_plan[subtask_num]
            env_idx = torch.where(self.subtask_pointer > subtask_num)[0]
            if isinstance(subtask, PlaceSubtask):
                subtask_progressive_success, subtask_progressive_checkers = (
                    self._place_check_success(
                        self.subtask_objs[subtask_num],
                        self.subtask_goals[subtask_num],
                        subtask.goal_rectangle_corners,
                        env_idx,
                        check_progressive_completion=True,
                    )
                )
            else:
                raise NotImplementedError(
                    f"{subtask.type} {type(subtask)} progressive completion checking not supported"
                )
            progressive_task_success[env_idx] &= subtask_progressive_success
            for k, v in subtask_progressive_checkers.items():
                new_k = f"{k}_progressive_{subtask_num}"
                if new_k not in progressive_task_checkers:
                    progressive_task_checkers[new_k] = torch.zeros(
                        self.num_envs, device=self.device, dtype=v.dtype
                    )
                progressive_task_checkers[new_k][env_idx] = v

        return progressive_task_success, progressive_task_checkers

    def _subtask_check_success(self):
        subtask_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        success_checkers = dict()

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )
        for subtask_num in currently_running_subtasks:
            subtask: Subtask = self.task_plan[subtask_num]
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            if isinstance(subtask, PickSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._pick_check_success(
                    self.subtask_objs[subtask_num],
                    env_idx,
                )
            elif isinstance(subtask, PlaceSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._place_check_success(
                    self.subtask_objs[subtask_num],
                    self.subtask_goals[subtask_num],
                    subtask.goal_rectangle_corners,
                    env_idx,
                )
            elif isinstance(subtask, NavigateSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._navigate_check_success(
                    self.subtask_objs[subtask_num],
                    self.subtask_goals[subtask_num],
                    (
                        subtask.articulation_config.articulation_type
                        if subtask.articulation_config is not None
                        else None
                    ),
                    env_idx,
                )
            elif isinstance(subtask, OpenSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._open_check_success(
                    self.subtask_articulations[subtask_num],
                    subtask.articulation_type,
                    subtask.articulation_handle_active_joint_idx,
                    subtask.articulation_handle_link_idx,
                    env_idx,
                )
            elif isinstance(subtask, CloseSubtask):
                (
                    subtask_success[env_idx],
                    subtask_success_checkers,
                ) = self._close_check_success(
                    self.subtask_articulations[subtask_num],
                    subtask.articulation_handle_active_joint_idx,
                    subtask.articulation_handle_link_idx,
                    env_idx,
                )
            else:
                raise NotImplementedError(
                    f"{subtask.type} {type(subtask)} not supported"
                )

            for k, v in subtask_success_checkers.items():
                if k not in success_checkers:
                    success_checkers[k] = torch.zeros(
                        self.num_envs, device=self.device, dtype=v.dtype
                    )
                success_checkers[k][env_idx] = v

        return subtask_success, success_checkers

    def _pick_check_success(
        self,
        obj: Actor,
        env_idx: torch.Tensor,
    ):
        is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.pick_cfg.ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 3:-2] - self.resting_qpos
        )
        robot_rest = torch.all(
            robot_rest_dist < self.pick_cfg.robot_resting_qpos_tolerance_grasping, dim=1
        )
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]
        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.pick_cfg.robot_cumulative_force_limit
        )
        subtask_checkers = dict(
            is_grasped=is_grasped,
            ee_rest=ee_rest,
            robot_rest=robot_rest,
            is_static=is_static,
            cumulative_force_within_limit=cumulative_force_within_limit,
        )
        if self._add_event_tracker_info:
            subtask_checkers["robot_target_pairwise_force"] = torch.norm(
                self.scene.get_pairwise_contact_forces(self.agent.finger1_link, obj)[
                    env_idx
                ],
                dim=1,
            ) + torch.norm(
                self.scene.get_pairwise_contact_forces(self.agent.finger2_link, obj)[
                    env_idx
                ],
                dim=1,
            )

        if self.detailed_success_checker_enable:
            ee_rest_err = torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            min_force, max_angle = get_grasp_force_angle(self.agent, obj)

            subtask_checkers.update(dict(
                ee_rest_err = ee_rest_err,
                min_force = min_force,
                max_angle = torch.rad2deg(max_angle),
            ))

        # print("============== success checker: ")
        # print(f"is_grasped: {is_grasped}")
        # print(f"ee_rest: {ee_rest}, ee_remain: {ee_remain.item():.2f}")
        # print(f"is_static: {is_static}")
        # print(f"cumulative_force_within_limit: {cumulative_force_within_limit}, force: {self.robot_cumulative_force[env_idx].item():.3f}")

        return (
            is_grasped
            & ee_rest
            & robot_rest
            & is_static
            & cumulative_force_within_limit,
            subtask_checkers,
        )

    def _place_check_success(
        self,
        obj: Actor,
        obj_goal: Actor,
        goal_rectangle_corners: torch.Tensor,
        env_idx: torch.Tensor,
        check_progressive_completion=False,
    ):
        is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]
        if self.place_cfg.goal_type == "zone":
            # (0 <= AM•AB <= AB•AB) and (0 <= AM•AD <=  AD•AD)
            As, Bs, Ds = (
                goal_rectangle_corners[env_idx, 0, :2],
                goal_rectangle_corners[env_idx, 1, :2],
                goal_rectangle_corners[env_idx, 3, :2],
            )
            Ms = obj.pose.p[env_idx, :2]

            AM = Ms - As
            AB = Bs - As
            AD = Ds - As

            AM_dot_AB = torch.sum(AM * AB, dim=1)
            AB_dot_AB = torch.sum(AB * AB, dim=1)
            AM_dot_AD = torch.sum(AM * AD, dim=1)
            AD_dot_AD = torch.sum(AD * AD, dim=1)

            xy_correct = (
                (0 <= AM_dot_AB)
                & (AM_dot_AB <= AB_dot_AB)
                & (0 <= AM_dot_AD)
                & (AM_dot_AD <= AD_dot_AD)
            )
            z_correct = (
                torch.abs(obj.pose.p[env_idx, 2] - obj_goal.pose.p[env_idx, 2])
                <= self.place_cfg.obj_goal_thresh
            )
            obj_at_goal = xy_correct & z_correct
        elif self.place_cfg.goal_type == "cylinder":
            xy_correct = (
                torch.norm(
                    obj.pose.p[env_idx, :2] - obj_goal.pose.p[env_idx, :2],
                    dim=1,
                )
                <= self.place_cfg.obj_goal_thresh
            )
            z_correct = (
                torch.abs(obj.pose.p[env_idx, 2] - obj_goal.pose.p[env_idx, 2])
                <= self.place_cfg.obj_goal_thresh
            )
            obj_at_goal = xy_correct & z_correct
        elif self.place_cfg.goal_type == "sphere":
            obj_at_goal = (
                torch.norm(
                    obj.pose.p[env_idx] - obj_goal.pose.p[env_idx],
                    dim=1,
                )
                <= self.place_cfg.obj_goal_thresh
            )
        else:
            raise NotImplementedError(
                f"{self.place_cfg.goal_type=} is not yet supported"
            )
        if check_progressive_completion:
            return obj_at_goal, dict(obj_at_goal=obj_at_goal)
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.place_cfg.ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 4:-2] - self.resting_qpos[1:]
        )
        robot_rest = torch.all(
            robot_rest_dist < self.place_cfg.robot_resting_qpos_tolerance, dim=1
        ) & (torch.abs(self.agent.robot.qpos[env_idx, 3] - self.resting_qpos[0]) < 0.01)
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]
        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.place_cfg.robot_cumulative_force_limit
        )
        subtask_checkers = dict(
            is_grasped=is_grasped,
            obj_at_goal=obj_at_goal,
            ee_rest=ee_rest,
            robot_rest=robot_rest,
            is_static=is_static,
            cumulative_force_within_limit=cumulative_force_within_limit,
        )
        if self._add_event_tracker_info:
            subtask_checkers["robot_target_pairwise_force"] = torch.norm(
                self.scene.get_pairwise_contact_forces(self.agent.finger1_link, obj)[
                    env_idx
                ],
                dim=1,
            ) + torch.norm(
                self.scene.get_pairwise_contact_forces(self.agent.finger2_link, obj)[
                    env_idx
                ],
                dim=1,
            )
        return (
            ~is_grasped
            & obj_at_goal
            & ee_rest
            & robot_rest
            & is_static
            & cumulative_force_within_limit,
            subtask_checkers,
        )

    def _is_grasping_partial_env_obj(obj, env_idx, max_angle=85):
        raise NotImplementedError()

    def _is_navigated_close(
        self,
        env_idx: torch.Tensor,
        goal: Actor,
        articulation_type: Optional[str] = None,
    ):
        navigated_close = (
            torch.norm(
                goal.pose.p[env_idx, :2] - self.agent.base_link.pose.p[env_idx, :2],
                dim=1,
            )
            <= self.navigate_cfg.navigated_successfully_dist
        )

        if isinstance(goal, Articulation) or isinstance(goal, Link):
            # NOTE (arth): assume nav to same articulation, since we check each parallel subtask at a time
            relative_pos_world = (
                self.agent.base_link.pose.p[env_idx] - goal.pose.p[env_idx]
            )

            relative_pos_local = quaternion_apply(
                quaternion_invert(goal.pose.q[env_idx]),
                relative_pos_world,
            )

            xrange = dict(fridge=[0.933, 1.833], kitchen_counter=[0.3, 1.5])[
                articulation_type
            ]
            yrange = dict(fridge=[-0.6, 0.6], kitchen_counter=[-0.6, 0.6])[
                articulation_type
            ]

            navigated_close &= (
                (xrange[0] <= relative_pos_local[:, 0])
                & (relative_pos_local[:, 0] <= xrange[1])
                & (yrange[0] <= relative_pos_local[:, 2])
                & (relative_pos_local[:, 2] <= yrange[1])
            )

        return navigated_close

    def _navigate_check_success(
        self,
        obj: Union[Actor, None],
        goal: Actor,
        articulation_type: str,
        env_idx: torch.Tensor,
    ):
        if obj is None:
            is_grasped = torch.zeros_like(env_idx, dtype=torch.bool)
        elif len(obj._scene_idxs) != self.num_envs:
            # NOTE (arth): this is so nav subtask train env can handle grasping when obj
            #   is only in some parallel envs -- not the cleanest implementation
            is_grasped = self._is_grasping_partial_env_obj(obj, env_idx, max_angle=30)
        else:
            is_grasped = self.agent.is_grasping(obj, max_angle=30)[env_idx]

        goal_pose_wrt_base = self.agent.base_link.pose.inv() * goal.pose
        targ = goal_pose_wrt_base.p[..., :2][env_idx]
        uc_targ = targ / torch.norm(targ, dim=1).unsqueeze(-1).expand(*targ.shape)
        rots = torch.sign(uc_targ[..., 1]) * torch.arccos(uc_targ[..., 0])
        oriented_correctly = (
            torch.abs(rots) <= self.navigate_cfg.navigated_successfully_rot
        )

        navigated_close = self._is_navigated_close(env_idx, goal, articulation_type)
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]

        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.navigate_cfg.robot_cumulative_force_limit
        )

        if self.navigate_cfg.ignore_arm_checkers:
            return (
                oriented_correctly
                & navigated_close
                & is_static
                & cumulative_force_within_limit,
                dict(
                    is_grasped=is_grasped,
                    oriented_correctly=oriented_correctly,
                    navigated_close=navigated_close,
                    is_static=is_static,
                    cumulative_force_within_limit=cumulative_force_within_limit,
                ),
            )

        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.navigate_cfg.ee_rest_thresh
        )
        if obj is None:
            robot_rest_dist = torch.abs(
                self.agent.robot.qpos[env_idx, 4:-2] - self.resting_qpos[1:]
            )
            robot_rest = torch.all(
                robot_rest_dist < self.navigate_cfg.robot_resting_qpos_tolerance, dim=1
            ) & (
                torch.abs(self.agent.robot.qpos[env_idx, 3] - self.resting_qpos[0])
                < 0.01
            )
        else:
            robot_rest_dist = torch.abs(
                self.agent.robot.qpos[env_idx, 3:-2] - self.resting_qpos
            )
            robot_rest = torch.all(
                robot_rest_dist
                < self.navigate_cfg.robot_resting_qpos_tolerance_grasping,
                dim=1,
            )
            if len(obj._scene_idxs) != self.num_envs:
                subtask_envs_with_obj = tensor_intersection_idx(
                    env_idx, obj._scene_idxs
                )
                robot_rest[subtask_envs_with_obj] &= torch.all(
                    robot_rest_dist[subtask_envs_with_obj]
                    < self.navigate_cfg.robot_resting_qpos_tolerance,
                    dim=1,
                )

        navigate_success = (
            oriented_correctly & navigated_close & ee_rest & robot_rest & is_static
        )
        if obj is not None:
            if len(obj._scene_idxs) != self.num_envs:
                subtask_envs_with_obj = tensor_intersection_idx(
                    env_idx, obj._scene_idxs
                )
                navigate_success[subtask_envs_with_obj] &= is_grasped[
                    subtask_envs_with_obj
                ]
            else:
                navigate_success &= is_grasped
        return (
            navigate_success & cumulative_force_within_limit,
            dict(
                is_grasped=is_grasped,
                oriented_correctly=oriented_correctly,
                navigated_close=navigated_close,
                ee_rest=ee_rest,
                robot_rest=robot_rest,
                is_static=is_static,
                cumulative_force_within_limit=cumulative_force_within_limit,
            ),
        )

    def _open_check_success(
        self,
        articulation: Articulation,
        articulation_type: str,
        active_joint_idx: int,
        link_idx: int,
        env_idx: torch.Tensor,
    ):
        is_grasped = self.agent.is_grasping(articulation.links[link_idx], max_angle=30)[
            env_idx
        ]
        articulation_open = articulation.qpos[env_idx, active_joint_idx] > (
            (
                articulation.qlimits[env_idx, active_joint_idx, 1]
                - articulation.qlimits[env_idx, active_joint_idx, 0]
            )
            * self.open_cfg.joint_qpos_open_thresh_frac[articulation_type]
            + articulation.qlimits[env_idx, active_joint_idx, 0]
        )
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.open_cfg.ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 4:-2] - self.resting_qpos[1:]
        )
        robot_rest = torch.all(
            robot_rest_dist < self.open_cfg.robot_resting_qpos_tolerance, dim=1
        ) & (torch.abs(self.agent.robot.qpos[env_idx, 3] - self.resting_qpos[0]) < 0.01)
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]
        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.open_cfg.robot_cumulative_force_limit
        )
        subtask_checkers = dict(
            is_grasped=is_grasped,
            articulation_open=articulation_open,
            ee_rest=ee_rest,
            robot_rest=robot_rest,
            is_static=is_static,
            cumulative_force_within_limit=cumulative_force_within_limit,
        )
        if self._add_event_tracker_info:
            subtask_checkers["handle_active_joint_qpos"] = articulation.qpos[
                env_idx, active_joint_idx
            ]
            subtask_checkers["handle_active_joint_qmin"] = articulation.qlimits[
                env_idx, active_joint_idx, 0
            ]
            subtask_checkers["handle_active_joint_qmax"] = articulation.qlimits[
                env_idx, active_joint_idx, 1
            ]
            subtask_checkers["robot_target_pairwise_force"] = torch.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger1_link, articulation.links[link_idx]
                )[env_idx],
                dim=1,
            ) + torch.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger2_link, articulation.links[link_idx]
                )[env_idx],
                dim=1,
            )
        return (
            articulation_open
            & ee_rest
            & robot_rest
            & is_static
            & cumulative_force_within_limit,
            subtask_checkers,
        )

    def _close_check_success(
        self,
        articulation: Articulation,
        active_joint_idx: int,
        link_idx: int,
        env_idx: torch.Tensor,
    ):
        is_grasped = self.agent.is_grasping(articulation.links[link_idx], max_angle=30)[
            env_idx
        ]
        articulation_closed = articulation.qpos[env_idx, active_joint_idx] < (
            (
                articulation.qlimits[env_idx, active_joint_idx, 1]
                - articulation.qlimits[env_idx, active_joint_idx, 0]
            )
            * self.close_cfg.joint_qpos_close_thresh_frac
            + articulation.qlimits[env_idx, active_joint_idx, 0]
        )
        ee_rest = (
            torch.norm(
                self.agent.tcp_pose.p[env_idx] - self.ee_rest_world_pose.p[env_idx],
                dim=1,
            )
            <= self.close_cfg.ee_rest_thresh
        )
        robot_rest_dist = torch.abs(
            self.agent.robot.qpos[env_idx, 4:-2] - self.resting_qpos[1:]
        )
        robot_rest = torch.all(
            robot_rest_dist < self.close_cfg.robot_resting_qpos_tolerance, dim=1
        ) & (torch.abs(self.agent.robot.qpos[env_idx, 3] - self.resting_qpos[0]) < 0.01)
        is_static = self.agent.is_static(threshold=0.2, base_threshold=0.05)[env_idx]
        cumulative_force_within_limit = (
            self.robot_cumulative_force[env_idx]
            < self.close_cfg.robot_cumulative_force_limit
        )
        subtask_checkers = dict(
            is_grasped=is_grasped,
            articulation_closed=articulation_closed,
            ee_rest=ee_rest,
            robot_rest=robot_rest,
            is_static=is_static,
            cumulative_force_within_limit=cumulative_force_within_limit,
        )
        if self._add_event_tracker_info:
            subtask_checkers["handle_active_joint_qpos"] = articulation.qpos[
                env_idx, active_joint_idx
            ]
            subtask_checkers["handle_active_joint_qmin"] = articulation.qlimits[
                env_idx, active_joint_idx, 0
            ]
            subtask_checkers["handle_active_joint_qmax"] = articulation.qlimits[
                env_idx, active_joint_idx, 1
            ]
            subtask_checkers["robot_target_pairwise_force"] = torch.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger1_link, articulation.links[link_idx]
                )[env_idx],
                dim=1,
            ) + torch.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger2_link, articulation.links[link_idx]
                )[env_idx],
                dim=1,
            )
        return (
            articulation_closed
            & ee_rest
            & robot_rest
            & is_static
            & cumulative_force_within_limit,
            subtask_checkers,
        )

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # OBS AND INFO
    # -------------------------------------------------------------------------------------------------

    def _get_obs_agent(self):
        agent_state = super()._get_obs_agent()
        agent_state["qpos"] = agent_state["qpos"][..., 3:]
        agent_state["qvel"] = agent_state["qvel"][..., 3:]
        return agent_state

    # NOTE (arth): for now, define keys that will always be added to obs. leave it to
    #       wrappers or task-specific envs to mask out unnecessary vals
    #       - subtasks that don't need that obs will set some default value
    #       - subtasks which need that obs will set value depending on subtask params
    def _get_obs_extra(self, info: Dict):
        base_pose_inv = self.agent.base_link.pose.inv()

        # all subtasks will have same computation for
        #       - tcp_pose_wrt_base :   tcp always there and is same link
        tcp_pose_wrt_base = vectorize_pose(base_pose_inv * self.agent.tcp.pose)

        #       - obj_pose_wrt_base :   different objs per subtask (or no obj)
        #       - goal_pos_wrt_base :   different goals per subtask (or no goal)
        obj_pose_wrt_base = torch.zeros(
            self.num_envs, 7, device=self.device, dtype=torch.float
        )
        goal_pos_wrt_base = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float
        )

        currently_running_subtasks = torch.unique(
            torch.clip(self.subtask_pointer, max=len(self.task_plan) - 1)
        )
        for subtask_num in currently_running_subtasks:
            env_idx = torch.where(self.subtask_pointer == subtask_num)[0]
            subtask = self.task_plan[subtask_num]
            if self.subtask_objs[subtask_num] is not None:
                if len(self.subtask_objs[subtask_num]._scene_idxs) != self.num_envs:
                    env_scene_idx = tensor_intersection(
                        env_idx, self.subtask_objs[subtask_num]._scene_idxs
                    )
                    obj_pose_wrt_base[env_scene_idx] = vectorize_pose(
                        base_pose_inv[env_scene_idx]
                        * self.subtask_objs[subtask_num].pose
                    )
                else:
                    obj_pose_wrt_base[env_idx] = vectorize_pose(
                        base_pose_inv * self.subtask_objs[subtask_num].pose
                    )[env_idx]
            if self.subtask_goals[subtask_num] is not None:
                if isinstance(subtask, OpenSubtask) or isinstance(
                    subtask, CloseSubtask
                ):
                    goal_pos_wrt_base[env_idx] = (
                        base_pose_inv * self.handle_world_poses[subtask_num]
                    ).p[env_idx]
                else:
                    goal_pos_wrt_base[env_idx] = (
                        base_pose_inv * self.subtask_goals[subtask_num].pose
                    ).p[env_idx]

        # already computed during evaluation is
        #       - is_grasped    :   part of success criteria (or set default)
        is_grasped = info["is_grasped"]

        origin_extra_obs = dict(
            tcp_pose_wrt_base=tcp_pose_wrt_base,
            obj_pose_wrt_base=obj_pose_wrt_base,
            goal_pos_wrt_base=goal_pos_wrt_base,
            is_grasped=is_grasped,
        )
    
        ###
        if self.pi0_info_merge_to_extra_obs and self.pi0_info_enable:
            if self.pi0_is_infer_mode:
                origin_extra_obs.update(dict(
                    pi0_eef_state = info["pi0_eef_state"],
                    # pi0_eef_ref_action = info["pi0_eef_ref_action"],
                    # pi0_eef_abs_action = info["pi0_eef_abs_action"],
                ))
            else:
                origin_extra_obs.update(dict(
                    pi0_eef_state = info["pi0_eef_state"],
                    pi0_eef_ref_action = info["pi0_eef_ref_action"],
                    pi0_eef_abs_action = info["pi0_eef_abs_action"],
                ))
        if self.policy_info_merge_to_extra_obs and self.policy_info_enable:
            origin_extra_obs.update(dict(
                qpos = info["qpos"],
                last_action = info["last_action"],

                head_camera_t = info["head_camera_t"],
                gripper_camera_t = info["gripper_camera_t"]
            ))

            if not self.policy_bc_policy_mode:
                origin_extra_obs.update(dict(
                    rot_error = info["rot_error"],
                    loc_error = info["loc_error"],

                    loc_diff = info["loc_diff"],
                    rotvec_diff = info["rotvec_diff"],

                    base_forward = info["base_forward"],
                    base2goal_vec = info["base2goal_vec"],
                ))

        return origin_extra_obs

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # REWARD (Ignored here)
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): this env does not have dense rewards since rewards are used for training subtasks.
    #       If need to train a subtask, extend this class to define a subtask
    # -------------------------------------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.subtask_pointer

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # -------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------
    # CAMERAS, SENSORS, AND RENDERING
    # -------------------------------------------------------------------------------------------------
    # NOTE (arth): also included the old "cameras" mode from MS2 since HAB renders this way
    # -------------------------------------------------------------------------------------------------

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        if self.render_mode == "human":
            room_camera_config = CameraConfig(
                "render_camera",
                sapien_utils.look_at([4, -3.5, 3.5], [1.5, -3.5, 0]),
                1920,
                1080,
                1,
                0.01,
                10,
            )
            return room_camera_config
        # this camera follows the robot around (though might be in walls if the space is cramped)
        robot_camera_pose = sapien_utils.look_at([-0.2, 0.5, 1], [0.2, -0.2, 0])
        robot_camera_config = CameraConfig(
            "render_camera",
            robot_camera_pose,
            512,
            512,
            1.75,
            0.01,
            10,
            mount=self.agent.torso_lift_link,
        )
        return robot_camera_config

    def set_moving_goal_poses_for_render(self):
        self.ee_rest_goal.set_pose(self.ee_rest_world_pose)
        for goal, pose in zip(self.subtask_goals, self.handle_world_poses):
            if pose is not None:
                goal.set_pose(pose)

    def render(self):
        self.set_moving_goal_poses_for_render()
        return super().render()
        

    # -------------------------------------------------------------------------------------------------
