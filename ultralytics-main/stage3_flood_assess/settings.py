
import os
from pathlib import Path

# ==================== Base paths ====================
# BASE_DIR = Path("/path/to/your/project/root")  # Example for user
BASE_DIR = None  # User should set their own project root
HIER_MULT_DIR = BASE_DIR / "stage3_flood_assess"

# Create necessary directories
(HIER_MULT_DIR / "data").mkdir(parents=True, exist_ok=True)
(HIER_MULT_DIR / "runs").mkdir(parents=True, exist_ok=True)
(HIER_MULT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

# ==================== Data paths ====================
# Example: set your own dataset paths below
DATA_ROOTS = {
    'front': None,  # User should set path to their own dataset
    'front_side': None,
    'rear': None,
    'rear_side': None,
    'side': None,
}

# ==================== Pretrained model paths (user should set) ====================
# Orientation classification model (YOLOv11n-cls) - consistent with training strategy (nano version)
# Path to orientation model, user should provide
ORIENTATION_MODEL = None

# Component detection models (YOLOv11n-det) - 5 expert models
# Path to component detection models, user should provide
COMPONENT_MODELS = {
    'front': None,
    'front_side': None,
    'rear': None,
    'rear_side': None,
    'side': None,
}

# ==================== Class definitions (strictly follow paper section 2.1.1 and 2.1.2) ====================

# Vehicle orientations (5 classes)
ORIENTATIONS = {
    0: 'front',         # 正前方
    1: 'front_side',    # 斜前方
    2: 'rear',          # 正后方
    3: 'rear_side',     # 斜后方
    4: 'side',          # 侧方
}
ORIENTATION_TO_ID = {v: k for k, v in ORIENTATIONS.items()}
NUM_ORIENTATIONS = 5


# 注意：内部索引为 0-4，显示值为 Level 1-5
FLOOD_GRADES = {
    0: 'extreme',    # Level 1: 极重度内涝 (>105cm) [内部索引0]
    1: 'severe',     # Level 2: 重度内涝 (75-105cm) [内部索引1]
    2: 'moderate',   # Level 3: 中度内涝 (35-75cm) [内部索引2]
    3: 'mild',       # Level 4: 轻度内涝 (0-35cm) [内部索引3]
    4: 'none',       # Level 5: 无内涝 (0cm) [内部索引4]
}

FLOOD_GRADES_CN = {
    0: '极重度内涝',  # 水位淹没车窗
    1: '重度内涝',    # 绝大部分车身组件消失
    2: '中度内涝',    # 水位达引擎进气口
    3: '轻度内涝',    # 积水遮挡保险杠/部分车轮
    4: '无内涝',      # 所有组件清晰可见
}
NUM_FLOOD_GRADES = 5

# ==================== 组件定义（严格按照论文2.1.2节）====================
# 每个朝向有4个特定的关键组件

COMPONENTS_PER_ORIENTATION = {
    # 前方 (Front): 4个组件
    'front': {
        0: 'front_glass',    # 前挡风玻璃
        1: 'hood',           # 发动机罩
        2: 'head_light',     # 前大灯
        3: 'front_bumper',   # 前保险杠
    },
    
    # 斜前方 (Front-Side): 4个组件
    'front_side': {
        0: 'front_glass',    # 前挡风玻璃
        1: 'side_glass',     # 侧窗玻璃
        2: 'head_light',     # 前大灯
        3: 'wheel',          # 车轮
    },
    
    # 后方 (Rear): 4个组件
    'rear': {
        0: 'rear_glass',     # 后挡风玻璃
        1: 'rear_light',     # 尾灯
        2: 'trunk_lid',      # 行李箱盖
        3: 'rear_bumper',    # 后保险杠
    },
    
    # 斜后方 (Rear-Side): 4个组件
    'rear_side': {
        0: 'side_glass',     # 侧窗玻璃
        1: 'rear_glass',     # 后挡风玻璃
        2: 'rear_light',     # 尾灯
        3: 'wheel',          # 车轮
    },
    
    # 侧方 (Side): 4个组件
    'side': {
        0: 'side_glass',     # 侧窗玻璃
        1: 'head_light',     # 前大灯
        2: 'wheel',          # 车轮
        3: 'rear_light',     # 尾灯
    },
}

# ==================== 洪水等级评估细则 ====================
# 详细的组件可见性规则，用于标注指导和可解释性分析

FLOOD_LEVEL_RULES = {
    'front': {
        # Front朝向的洪水等级评估规则
        # 组件：0:front_glass, 1:hood, 2:head_light, 3:front_bumper
        # 注意：字典键为内部索引 0-4，显示值为 Level 1-5
        4: {  # Level 5: 无内涝 [内部索引4]
            'description': '地面无积水，所有组件均可完整检测到',
            'visible_components': [0, 1, 2, 3],  # 所有组件可见
            'partially_visible': [],
            'invisible_components': [],
        },
        3: {  # Level 4: 轻度内涝 [内部索引3]
            'description': '地面有积水，front_bumper下缘被淹没1/3',
            'visible_components': [0, 1, 2],  # front_glass, hood, head_light
            'partially_visible': [3],  # front_bumper部分可见
            'invisible_components': [],
        },
        2: {  # Level 3: 中度内涝 [内部索引2]
            'description': 'front_bumper完全不可见',
            'visible_components': [0, 1, 2],  # front_glass, hood, head_light
            'partially_visible': [],
            'invisible_components': [3],  # front_bumper不可见
        },
        1: {  # Level 2: 重度内涝 [内部索引1]
            'description': 'front_bumper, head_light, hood完全不可见',
            'visible_components': [0],  # 仅front_glass可见
            'partially_visible': [],
            'invisible_components': [1, 2, 3],
        },
        0: {  # Level 1: 极重度内涝 [内部索引0]
            'description': 'front_glass部分可见或完全不可见',
            'visible_components': [],
            'partially_visible': [0],  # front_glass可能部分可见
            'invisible_components': [1, 2, 3],
        },
    },
    
    'rear': {
        # Rear朝向的洪水等级评估规则
        # 组件：0:rear_glass, 1:rear_light, 2:trunk_lid, 3:rear_bumper
        4: {
            'description': '地面无大范围积水，所有组件均可完整检测到',
            'visible_components': [0, 1, 2, 3],
            'partially_visible': [],
            'invisible_components': [],
        },
        3: {
            'description': 'rear_bumper下缘被淹没1/3',
            'visible_components': [0, 1, 2],
            'partially_visible': [3],
            'invisible_components': [],
        },
        2: {
            'description': 'rear_bumper完全不可见',
            'visible_components': [0, 1, 2],
            'partially_visible': [],
            'invisible_components': [3],
        },
        1: {
            'description': 'rear_bumper, trunk_lid, rear_light完全不可见',
            'visible_components': [0],
            'partially_visible': [],
            'invisible_components': [1, 2, 3],
        },
        0: {
            'description': 'rear_glass部分可见或完全不可见',
            'visible_components': [],
            'partially_visible': [0],
            'invisible_components': [1, 2, 3],
        },
    },
    
    'front_side': {
        # Front_side朝向的洪水等级评估规则
        # 组件：0:front_glass, 1:side_glass, 2:head_light, 3:wheel
        4: {
            'description': '地面无大范围积水，所有组件均可完整检测到',
            'visible_components': [0, 1, 2, 3],
            'partially_visible': [],
            'invisible_components': [],
        },
        3: {
            'description': 'wheel被淹没1/2',
            'visible_components': [0, 1, 2],
            'partially_visible': [3],
            'invisible_components': [],
        },
        2: {
            'description': 'wheel完全不可见，head_light可见/不可见',
            'visible_components': [0, 1],  # front_glass, side_glass
            'partially_visible': [2],  # head_light可能可见
            'invisible_components': [3],
        },
        1: {
            'description': 'wheel, head_light完全不可见',
            'visible_components': [0, 1],
            'partially_visible': [],
            'invisible_components': [2, 3],
        },
        0: {
            'description': 'front_glass部分可见或完全不可见',
            'visible_components': [1],  # 可能仅side_glass可见
            'partially_visible': [0],
            'invisible_components': [2, 3],
        },
    },
    
    'rear_side': {
        # Rear_side朝向的洪水等级评估规则
        # 组件：0:side_glass, 1:rear_glass, 2:rear_light, 3:wheel
        4: {
            'description': '地面无积水，所有组件均可完整检测到',
            'visible_components': [0, 1, 2, 3],
            'partially_visible': [],
            'invisible_components': [],
        },
        3: {
            'description': 'wheel被淹没1/2',
            'visible_components': [0, 1, 2],
            'partially_visible': [3],
            'invisible_components': [],
        },
        2: {
            'description': 'wheel完全不可见',
            'visible_components': [0, 1, 2],
            'partially_visible': [],
            'invisible_components': [3],
        },
        1: {
            'description': 'wheel, rear_light完全不可见',
            'visible_components': [0, 1],  # side_glass, rear_glass
            'partially_visible': [],
            'invisible_components': [2, 3],
        },
        0: {
            'description': 'rear_glass部分可见或完全不可见',
            'visible_components': [0],  # 可能仅side_glass可见
            'partially_visible': [1],
            'invisible_components': [2, 3],
        },
    },
    
    'side': {
        # Side朝向的洪水等级评估规则
        # 组件：0:side_glass, 1:head_light, 2:wheel, 3:rear_light
        4: {
            'description': '地面无积水，所有组件均可完整检测到',
            'visible_components': [0, 1, 2, 3],
            'partially_visible': [],
            'invisible_components': [],
        },
        3: {
            'description': 'wheel被淹没1/2',
            'visible_components': [0, 1, 3],
            'partially_visible': [2],
            'invisible_components': [],
        },
        2: {
            'description': 'wheel完全不可见，head_light可见/不可见',
            'visible_components': [0, 3],
            'partially_visible': [1],
            'invisible_components': [2],
        },
        1: {
            'description': 'wheel, head_light, rear_light完全不可见',
            'visible_components': [0],
            'partially_visible': [],
            'invisible_components': [1, 2, 3],
        },
        0: {
            'description': 'side_glass部分可见或完全不可见',
            'visible_components': [],
            'partially_visible': [0],
            'invisible_components': [1, 2, 3],
        },
    },
}

NUM_COMPONENTS_PER_ORIENTATION = 4  # 每个朝向固定4个组件
NUM_COMPONENTS = 4  # 简写，与上面相同

# 组件中文名称（用于可视化）
COMPONENT_NAMES_CN = {
    'front_glass': '前挡风玻璃',
    'rear_glass': '后挡风玻璃',
    'side_glass': '侧窗玻璃',
    'hood': '发动机罩',
    'trunk_lid': '行李箱盖',
    'head_light': '前大灯',
    'rear_light': '尾灯',
    'front_bumper': '前保险杠',
    'rear_bumper': '后保险杠',
    'wheel': '车轮',
}

