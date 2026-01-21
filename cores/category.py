from enum import Enum


class Category(Enum):
    STATIC_OBJECT = 0
    VEHICLE = 1
    PEOPLE = 2
    ROAD = 3
    FREE = 99
    UNKNOWN = 100
    
    @classmethod
    def get_name(cls, value):
        """根据值获取类别名"""
        return cls(value).name
    
    @classmethod
    def get_all_names(cls):
        """获取所有类别名"""
        return [member.name for member in cls]
    
    @classmethod
    def get_name_value_dict(cls):
        """获取名字到值的字典"""
        return {member.name: member.value for member in cls}
    

# # 使用示例
# # 1. 获取单个名字
# car_item = Category.CAR
# print(f"枚举项: {car_item}")           # Category.CAR
# print(f"类别名: {car_item.name}")      # CAR
# print(f"对应值: {car_item.value}")     # 1

# # 2. 根据值获取名字
# print(Category.get_name(1))          # CAR
# print(Category(2).name)              # PEOPLE

# # 3. 获取所有名字
# print(Category.get_all_names())      # ['CAR', 'PEOPLE', 'ANIMAL', 'BUILDING']

# # 4. 获取映射字典
# print(Category.get_name_value_dict())  # {'CAR': 1, 'PEOPLE': 2, ...}