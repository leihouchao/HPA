####################################################
# 
# design the app
# 
####################################################
from tkinter import *
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tkinter import ttk
from tkinter import filedialog
import tkinter.messagebox
import tkinter
import caffe
import numpy as np 
import matplotlib.pyplot as plt;

def sigmoid(x):
    return 1./(1 + np.exp(-x))

# 对一张图片进行预测
def predict(file_name):
    # 1,定义网络
    net = caffe.Net("./model/test_app.prototxt", "./saved_model/solver_iter_82000.caffemodel", caffe.TEST)

    # 2,传入数据
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
    transformer.set_transpose('data', (2,0,1))                                  #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
    transformer.set_raw_scale('data', 255)                                      # 缩放到【0，255】之间
    transformer.set_channel_swap('data', (2,1,0))                               #交换通道，将图片由RGB变为BGR

    im = caffe.io.load_image(file_name)                                       #加载图片
    net.blobs['data'].data[...] = transformer.preprocess('data',im)           #执行上面设置的图片预处理操作，并将图片载入到blob中

    # 3,前向计算
    net.forward()

    # 4,预测
    pred = net.blobs['score'].data
    pred = sigmoid(pred)
    return pred

# 根据预测结果,可视化图片的推断结果
def visualize(file_name, pred):

    # 1,id与名字的对应
    id_2_label = [
        "Mitochondria", "Nucleus", "Endoplasmic reticulum", "Nuclear speckles", 
        "Plasma membrane", "Nucleoplasm", "Cytosol", "Nucleoli",
        "Vesicles", "Golgi apparatus"
    ]
 
    # 2,读取图片
    setFont = ImageFont.truetype('./font.ttf', 50)
    fillColor = "#fff"
    im = Image.open(file_name)
    draw = ImageDraw.Draw(im)

    top1 = np.argmax(pred)
    label =  "%s : %.2f%%" % (id_2_label[top1], float(pred[top1]) * 100)
    pred[top1] = 0
    draw.text(xy = (50, 20), text = label, font=setFont, fill=fillColor)

    top2 = np.argmax(pred)
    label =  "%s : %.2f%%" % (id_2_label[top2], float(pred[top2]) * 100)
    pred[top2] = 0
    draw.text(xy = (50, 120), text = label, font=setFont, fill=fillColor)

    top3 = np.argmax(pred)
    label =  "%s : %.2f%%" % (id_2_label[top3], float(pred[top3]) * 100)
    pred[top3] = 0
    draw.text(xy = (50, 220), text = label, font=setFont, fill=fillColor)
 
    # 图片保存
    img_name = (app.selected_img.strip().split("/"))[-1]
    im.save("./app_result/" + img_name)
   
def app():
    # 主窗口
    root = Tk()

    #  标题
    root.title("上海交通大学计算机系 ～ hpa多标签分类")

    #  尺寸
    root.geometry('700x750')

    # 显示图片
    img_open = Image.open('bg.png')
    img_open = img_open.resize((650, 650))
    ori_img = ImageTk.PhotoImage(img_open)
    label_img = Label(root, image = ori_img, width = 700, height = 700)
    label_img.place(x = 0, y = 0)

    # 选择的图片
    selected_img = ""

    # 打开文件按钮
    def open_callback():
        app.selected_img = filedialog.askopenfilename(title='打开jpeg文件', filetypes=[('All Files', '*')])
        print("选择的图片是: ", app.selected_img)

        if(app.selected_img == "" or app.selected_img == ()):
            tkinter.messagebox.showwarning('警告','请选择文件')
        else:
            img_open = Image.open(app.selected_img)
            img_open = img_open.resize((650, 650))
            other_img = ImageTk.PhotoImage(img_open)
            label_img.config(image = other_img)
            label_img.image= other_img

    open_btn = Button(root, text ="打开文件", font = ('微软雅黑',12), command = open_callback)
    open_btn.place(x = 450, y = 700)

    # 开始预测按钮
    def predict_callback():
        print("开始预测")

        if(app.selected_img == ""):
            # print("请选择图片")
            tkinter.messagebox.showwarning('警告','请选择文件')
        else:

            tkinter.messagebox.showinfo('提示','开始模型推断...')
            # 进行预测
            pred = predict(app.selected_img)[0]
            print("预测的结果是: ", pred)
            tkinter.messagebox.showinfo('提示','模型推断完成...')
            
            # 可视化预测结果
            visualize(app.selected_img, pred)

            # 展示预测的结果
            img_name = (app.selected_img.strip().split("/"))[-1]
            img_open = Image.open("./app_result/" + img_name)
            img_open = img_open.resize((650, 650))
            other_img = ImageTk.PhotoImage(img_open)
            label_img.config(image = other_img)
            label_img.image= other_img
            app.selected_img = ""

    predict_btn = Button(root, text ="开始预测", font = ('微软雅黑',12), command = predict_callback)
    predict_btn.place(x = 550, y = 700)


    # 开始显示
    root.mainloop()

app()