import struct

def read_pcd_binary(file_path):
    """
    从二进制PCD文件中读取点云数据
    """
    # 读取文件头
    with open(file_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header_lines.append(line)
            if line.startswith('DATA'):
                break
        
        # 解析文件头信息
        header_info = {}
        for line in header_lines:
            if line.startswith('FIELDS'):
                header_info['fields'] = line.split()[1:]
            elif line.startswith('SIZE'):
                header_info['sizes'] = list(map(int, line.split()[1:]))
            elif line.startswith('TYPE'):
                header_info['types'] = line.split()[1:]
            elif line.startswith('COUNT'):
                header_info['counts'] = list(map(int, line.split()[1:]))
            elif line.startswith('WIDTH'):
                header_info['width'] = int(line.split()[1])
            elif line.startswith('HEIGHT'):
                header_info['height'] = int(line.split()[1])
            elif line.startswith('POINTS'):
                header_info['points'] = int(line.split()[1])
            elif line.startswith('DATA'):
                header_info['data_type'] = line.split()[1]
    
    # print("文件头信息:")
    # for key, value in header_info.items():
    #     print(f"  {key}: {value}")
    
    # 计算每个点的字节数
    point_size = sum(header_info['sizes'])
    total_points = header_info['points']
    
    # 读取二进制数据
    with open(file_path, 'rb') as f:
        # 跳过文件头
        while True:
            line = f.readline().decode('utf-8').strip()
            if line.startswith('DATA'):
                break
        
        # 读取所有点数据
        binary_data = f.read()
    
    # 解析二进制数据
    points_data = []
    for i in range(total_points):
        start_idx = i * point_size
        end_idx = start_idx + point_size
        
        if end_idx > len(binary_data):
            break
            
        point_bytes = binary_data[start_idx:end_idx]
        
        # 根据字段解析数据
        offset = 0
        point_dict = {}
        
        for field, size, field_type in zip(header_info['fields'], 
                                         header_info['sizes'], 
                                         header_info['types']):
            field_bytes = point_bytes[offset:offset+size]
            
            if field_type == 'F' and size == 4:  # 32位浮点数
                value = struct.unpack('f', field_bytes)[0]
            elif field_type == 'F' and size == 8:  # 64位浮点数
                value = struct.unpack('d', field_bytes)[0]
            elif field_type == 'U' and size == 1:  # 8位无符号整数
                value = struct.unpack('B', field_bytes)[0]
            elif field_type == 'I' and size == 4:  # 32位整数
                value = struct.unpack('i', field_bytes)[0]
            else:
                value = 0
                
            point_dict[field] = value
            offset += size
            
        points_data.append(point_dict)
    
    # 转换为numpy数组
    fields = header_info['fields'][:4]
    point_cloud = np.array([[point[field] for field in fields] for point in points_data])
    
    # print(f"\n成功读取 {len(point_cloud)} 个点")
    # print(f"数据形状: {point_cloud.shape}")
    # print(f"字段: {fields}")
    
    return point_cloud, header_info