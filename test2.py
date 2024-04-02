# 假设这是你的 results_list2
results_list2 = [
    [('猫', 90.5), ('狗', 5.4), ('鸟', 2.0)],
    [('苹果', 80.0), ('香蕉', 15.0), ('橙子', 5.0)],
    [('狗', 100.0)],
    [('狗', 100.0)]
    # 其他图片的结果...
]

# 创建一个字典来存储整合后的结果
integrated_results = {}

# 遍历 results_list2 中的每个图片的结果列表
for image_results in results_list2:
    # 创建一个临时字典来存储当前图片的结果
    temp_results = {}

    # 遍历当前图片的结果元组
    for class_name, probability in image_results:
        if class_name not in temp_results:
            temp_results[class_name] = probability
        else:
            temp_results[class_name] += probability

    # 将当前图片的结果添加到整合的结果中
    for class_name, probability in temp_results.items():
        if class_name not in integrated_results:
            integrated_results[class_name] = probability
        else:
            integrated_results[class_name] += probability

print(integrated_results)
