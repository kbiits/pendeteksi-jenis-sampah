import streamlit as st
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

model = torch.load("final_model.pt")
model.eval()

transformer = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225]
                                                       )])


def transform_img(image):
    return transformer(image).unsqueeze(0)


classes = ['Organik', 'Anorganik']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_available = torch.cuda.is_available()


def predict(img):
    img = Image.open(io.BytesIO(img))
    tensor = transform_img(img)
    tensor = tensor.to(device)
    output = model.forward(tensor)

    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not gpu_available else np.squeeze(
        preds_tensor.cpu().numpy())
    return classes[preds]


st.title("Klasifikasi Jenis Sampah")


st.markdown("<h4 style='text-align: center;'>Silahkan foto sampah anda</h4>", unsafe_allow_html=True)
input_img = st.camera_input("")
st.markdown("<h4 style='text-align: center;'>Atau upload disini</h4>", unsafe_allow_html=True)
input_img_2 = st.file_uploader("Pilih foto")

final_img = input_img if input_img is not None else input_img_2

if final_img is not None:
    result = predict(final_img.read())
    with st.columns(3)[1]:
        st.image(final_img) 
    st.markdown(f"<h4 style='text-align: center;'>Sampah anda berjenis</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: green; background: white; font-size: 2rem;'>{result}</h4>", unsafe_allow_html=True)
