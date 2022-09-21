import torch
import sys
import tkinter as tk


window = tk.Tk()
window.title(string="Model Merger")
tk.Label(text = "Model Merger",font=("Arial",25)).pack()
tk.Label(text = "GUI by antrobot1234").pack()

frame1 = tk.Frame()
frame2 = tk.Frame()
frame3 = tk.Frame()

frameSlider = tk.Frame()
frameButton = tk.Frame()

tk.Label(frame1,text = "File 1:").pack(side="left")
file1text = tk.Entry(frame1,width=40)
file1text.pack(side="left")

tk.Label(frame2,text = "File 2:").pack(side="left")
file2text = tk.Entry(frame2,width=40)
file2text.pack(side="left")

tk.Label(frame3,text = "File Out:").pack(side="left")
fileOtext = tk.Entry(frame3,width=38)
fileOtext.pack(side="left")

tk.Label(frameSlider,text = "Weight of file 1").pack(side="left")
scale = tk.Scale(frameSlider,from_=0, to=100,orient="horizontal",tickinterval=10,length=450)
scale.pack(side="left")



goButton = tk.Button(frameButton,text="RUN",height=2,width=20,bg="green")
def merge(file1,file2,out,a):
    alpha = (a)/100
    if not(file1.endswith(".ckpt")):
        file1 += ".ckpt"
    if not(file2.endswith(".ckpt")):
        file2 += ".ckpt"
    if not(out.endswith(".ckpt")):
        out += ".ckpt"
    #Load Models
    model_0 = torch.load(file1)
    model_1 = torch.load(file2)
    theta_0 = model_0['state_dict']
    theta_1 = model_1['state_dict']

    for key in theta_0.keys():
        if 'model' in key and key in theta_1:
            theta_0[key] = (alpha) * theta_0[key] + (1-alpha) * theta_1[key]

    goButton.config(bg="red",text="RUNNING...\n(STAGE 2)")
    window.update()

    for key in theta_1.keys():
        if 'model' in key and key not in theta_0:
            theta_0[key] = theta_1[key]
    torch.save(model_0, out)
    

def handleClick(event):
    goButton.config(bg="red",text="RUNNING...\n(STAGE 1)")
    window.update()
    merge(file1text.get(),file2text.get(),fileOtext.get(),scale.get())
    goButton.config(bg="green",text="RUN")
    
goButton.pack()
goButton.bind("<Button-1>",handleClick)


frame1.pack()
frame2.pack()
frame3.pack()
frameSlider.pack()
frameButton.pack()

window.mainloop()
