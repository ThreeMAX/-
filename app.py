from flask import Flask, make_response, render_template, request, jsonify
from static.model_fenlei.model import efficientnet_b0 as create_model
import torch
from torchvision import transforms
import json
from PIL import Image
from static.model_fenge.unet import Unet
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from flask import url_for

# EfficientNet模型代码植入
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

img_size = {"B0": 224,
            "B1": 240,
            "B2": 260,
            "B3": 300,
            "B4": 380,
            "B5": 456,
            "B6": 528,
            "B7": 600}

num_model = "B0"

json_path = './static/model_fenlei/class_indices_butterfly.json'

with open(json_path, "r") as f:
    class_indict = json.load(f)

model = create_model(num_classes=100).to(device)
model_weight_path = "./static/model_fenlei/best_model_loss_butterfly.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# U-Net模型代码植入
unet = Unet()
count = False
name_classes = ["background", "蝴蝶"]

app = Flask(__name__)

@app.route('/', methods=['GET'])
#  装饰器，修饰我们底下的函数，让我们的函数具备下面的功能
#  程序的路由，当浏览器访问我们的根路径的时候，会执行下面的函数
def main_page():  # put application's code here
    res = make_response(render_template('log_in.html'))
    return res

@app.route('/login', methods=['POST', 'GET'])

def login():

    if request.method == 'POST':
        name = request.form["user"]
        password = request.form["password"]
        if name == 'SUJUNYI' and password == '12345':
            return render_template("classifier.html")
        # 核实密码以及用户名
        else:
            return render_template("404.html")
    else:
        return render_template("404.html")

# 2023年9月2号添加，为了显示预测最大值的病媒生物的信息
def find_info_by_name(excel_file, name_to_find):
    try:
        # 打开Excel文件
        workbook = openpyxl.load_workbook(excel_file)
        sheet = workbook.active

        # 初始化一个列表来存储匹配的信息
        info_list = []

        # 遍历每一行，查找匹配的名称并将信息添加到列表中
        for row in sheet.iter_rows(values_only=True):
            if row[0] == name_to_find:
                info = [row[0], row[1]]
                info_list.append(info)

        # 关闭Excel文件
        workbook.close()

        return info_list

    except Exception as e:
        print("发生错误:", e)
        return []

# 替换成你的Excel文件路径
excel_file = "E:\Web开发\病媒识别4\病媒生物介绍.xlsx"

@app.route('/predict', methods=['GET', 'POST'])
# 后端图片上传
def predict():

    ########################预测多张，并且把每一张的概率都给打印在网页上###############################
    if request.method == 'GET':
        res = make_response(render_template('classifier.html'))
        return res

    elif request.method == 'POST':
        if 'myImg' in request.files:
            uploaded_files = request.files.getlist('myImg')  # Get list of uploaded files

            results_list = []  # List to store results for each image
            results_list2 = []
            integrated_results = {}
            segmented_image_urls = []

            for objFile in uploaded_files:
                strFileName = objFile.filename
                strFilePath = "./static/myimages/" + save_time + strFileName
                objFile.save(strFilePath)
                strFilePath2 = "./static/myimages_fengehou/" + save_time + strFileName

                segmented_image_url = url_for('static', filename='myimages_fengehou/' + save_time + strFileName)
                segmented_image_urls.append(segmented_image_url)

                # 上面这部分属于对于传入的图片的保存
                # 下面这部分属于对于保存的图片进行图像分割，再将其传入到分类模型得出结果

                img = Image.open(strFilePath)
                r_image = unet.detect_image(img, count=count, name_classes=name_classes)
                r_image.save(strFilePath2)


                img = Image.open(strFilePath2)
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)

                with torch.no_grad():
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    print(predict.shape)
                    predict_topk_values, predict_topk_indices = torch.topk(predict, k=5, dim=0)

                predict_topk_indices = predict_topk_indices.cpu().numpy()  # Convert tensor indices to numpy array

                predict_class_names = [class_indict[str(idx.item())] for idx in predict_topk_indices]
                predict_probabilities = predict_topk_values.numpy()

                # results系列的代码，把每一个图片的前五种都显示出来，然后拼到results_list当中
                # results2系列的代码，把每一张图片的前五种概率进行整合，然后拼接得出所有的平均概率，然后拼到results_list2当中

                results = []
                results2 = []

                for class_name, probability in zip(predict_class_names, predict_probabilities):
                    percent_probability = probability * 100
                    result = "  {}的概率大小为: {:.1f}%".format(class_name, percent_probability)
                    results.append(result)

                    result2 = (class_name, percent_probability)
                    results2.append(result2)


                results_list.append(results)  # Store results for current image
                results_list2.append(results2)

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

                # 将当前图片的结果添加到整合的结果中，就是把每张图片的类别概率进行相加，如果没有这个类别就添加这个类别
                for class_name, probability in temp_results.items():
                    if class_name not in integrated_results:
                        integrated_results[class_name] = probability
                    else:
                        integrated_results[class_name] += probability

            # print(integrated_results)

            # 将字典中的值转换为列表
            values = np.array(list(integrated_results.values()))
            # print(values)

            # 各个值除以，总的值
            normalized_values = values / sum(values) * 100
            # print(normalized_values)

            # 创建一个新的字典，包含归一化后的结果，就是加起来是100
            normalized_results = {class_name: probability for class_name, probability in
                                  zip(integrated_results.keys(), normalized_values)}
            # print(normalized_results)

            # 然后进行从大到小的排序，并且把每个概率是多少拼接到一个列表当中

            out_list = []
            out_list2 = []
            sorted_results = dict(sorted(normalized_results.items(), key=lambda item: item[1], reverse=True))
            print(sorted_results)
            for class_name, probability in sorted_results.items():
                average_probability = probability
                # print("类别 '{}' 的平均概率是 {:.2f}%".format(class_name, average_probability))
                out_list.append("类别 '{}' 的平均概率是 {:.2f}%".format(class_name, average_probability))
                out_list2.append(class_name)

                # print(out_list)

            print(out_list[0])
            print(out_list2[0])

            find_name = out_list2[0]

            xinxi = find_info_by_name(excel_file, find_name)
            if xinxi:
                for info in xinxi:
                    out_list.append(info[1])
                    # out_list.append(info[0])
            print(xinxi)
            print(out_list)
            print(out_list2)

            # 返回预测结果列表
            return jsonify(results=out_list, segmented_images=segmented_image_urls)

        else:
            err = "error"
            return err
    else:
        err = "error"
        return err

@app.route('/classifier.html')
def click1():
    res = make_response(render_template('classifier.html'))
    return res

picFolder = os.path.join('static', '大头金蝇2')
app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/database.html')
def click2():
    imageList = os.listdir('static/大头金蝇2')
    imagelist = ['大头金蝇2/' + image for image in imageList]
    return render_template('database.html', imagelist=imagelist)
    # res = make_response(render_template('database.html'))
    # return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# http 80 https 443 tcp/ip
# get post request response

# get request -> 服务器处理 -> response -> 浏览器渲染html内容
# html 是超文本标记语言 js css jqurey