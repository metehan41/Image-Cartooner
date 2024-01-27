import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
import numpy as np


def read_image_cv2(file_path):
    try:
        img = cv2.imread(file_path)
        if img is not None:
            return img
        else:
            print("Failed to read the image. Please provide a valid file path.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def GaussianFilterFuncimage(image, sigma, img_name):
  print(f"************ Gaussian Filtered Image With {sigma}************")
  smoothed_imaged = gaussian_filter(image, sigma)
  cv2.imwrite('report/result/'+img_name + "_GaussianFltered sigma_" +str(sigma) + ".jpg",smoothed_imaged)
  return smoothed_imaged


def MedianFilterFunc(image, kernel_size, img_name):
    print(f"************ Median Filtered Image with {kernel_size}************")
    # Apply the median filter
    median_filtered_image = median_filter(image, kernel_size)
    # Display the original and filtered images
    cv2.imwrite('report/result/'+img_name + "_MedianFiltered kernel_" +str(kernel_size) + ".jpg",median_filtered_image)

    return median_filtered_image


def DoG_filter_func(image, parameters, img_name):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Apply Gaussian filter with sigma1
  gaussian_filtered_image_1 = gaussian_filter(gray, sigma= parameters['sigma'])
  # Apply Gaussian filter with sigma2
  gaussian_filtered_image_2 = gaussian_filter(gray, sigma= parameters['sigma'] * parameters['k'])
  # Calculate DoG filter
  DoG_filtered_image = gaussian_filtered_image_1 - gaussian_filtered_image_2
  # Threshold the filtered image
  thresholded_image = np.where(DoG_filtered_image > parameters['threshold'], 1, 0)
  print(f"DoG Filtered Image with sigma {parameters['sigma']}, k {parameters['k']}, threshold {parameters['threshold']}")
  cv2.imwrite('report/result/'+img_name + "EdgeDetected Image parameters" +str(parameters) + ".jpg",thresholded_image*255)

  return thresholded_image


def quantizer(image, quantization_level, img_name):
  # Convert image to Lab color space
  input_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
  # Quantize the L channel (Luminance)
  quantized_lab = input_lab.copy()
  quantized_lab[:,:, 0] = (quantized_lab[:,:, 0] // quantization_level) * quantization_level  # Adjust the quantization level, when increase to 10 there is bordes in color of sky
  # Convert back to BGR
  quantized_image = cv2.cvtColor(quantized_lab, cv2.COLOR_Lab2BGR)
  print(f"quantized image with {quantization_level} level")
  cv2.imwrite('report/result/'+img_name + "Quantized Image quantization level" +str(quantization_level) + ".jpg",quantized_image)

  return quantized_image



def combinator(quantized_image, inverted_edges, img_name):

  b, g, r = cv2.split(quantized_image)
  r_result = cv2.multiply(r, inverted_edges, dtype=cv2.CV_32F)
  g_result = cv2.multiply(g, inverted_edges, dtype=cv2.CV_32F)
  b_result = cv2.multiply(b, inverted_edges, dtype=cv2.CV_32F)
  combined_image = cv2.merge((b_result, g_result, r_result))
  print("Final Cartooned Image")
  cv2.imwrite('report/result/'+img_name + "Cartoon Image" + ".jpg",combined_image)

  return combined_image


def cartooner(img, edge, smoothed_image, parameters, quantization_level, img_name):

  # Inversion (binary)
  inverted_edges = 1 - edge
  quantized_image = quantizer(smoothed_image, quantization_level, img_name)
  cartoon_image = combinator(quantized_image, inverted_edges, img_name + "quantized_" + str(quantization_level))
  return cartoon_image


def main():

  img_name = "image1"
  img_1_path = 'report/data/galata-tower-2-1210325.jpg'
  img1 = read_image_cv2(img_1_path)
  cv2.imwrite('report/result/'+img_name + "_orginal"+".jpg",img1)
  
  sigmas = [(1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0), (5, 5, 0)]
  for i in sigmas:
    GaussianFilterFuncimage(img1, i, img_name)

  kernels = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]
  for i in kernels:
    MedianFilterFunc(img1, i, img_name)

  smoothed_image = gaussian_filter(img1, (2, 2, 0))
  parameters = {'sigma': 1.4, 'k':1.1, "threshold": 30} #
  edge = DoG_filter_func(smoothed_image, parameters, img_name)

  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 30} #
  edge = DoG_filter_func(smoothed_image, parameters, img_name)

  parameters = {'sigma': 2.2, 'k':1.1, "threshold": 30} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters, img_name)
  
  parameters = {'sigma': 2.6, 'k':1.1, "threshold": 30} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)

  ##########################################################################

  quantization_level = 4
  parameters = {'sigma': 1.8, 'k':1.05, "threshold": 30} #
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" + name)

  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 30} #
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1.2, "threshold": 30} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1.3, "threshold": 30} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1.4, "threshold": 30} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1.7, "threshold": 30} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':2, "threshold": 30} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':3, "threshold": 30} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  ##########################################################################

  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 0} #
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 20} #
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 30} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 50} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 60} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 80} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1, "threshold": 100} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  parameters = {'sigma': 1.8, 'k':1, "threshold": 200} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)
  name = str(parameters)
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "DoG" +  name)

  ##########################################################################

  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 30} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name)

  inverted_edges = 1 - edge

  quantization_level = 4
  quantized_image = quantizer(smoothed_image, quantization_level, img_name)
  cartoon_image = combinator(quantized_image, inverted_edges, img_name + "quantized_" + str(quantization_level))

  quantization_level = 6
  quantized_image = quantizer(smoothed_image, quantization_level, img_name)
  cartoon_image = combinator(quantized_image, inverted_edges, img_name + "quantized_" + str(quantization_level))

  quantization_level = 8
  quantized_image = quantizer(smoothed_image, quantization_level, img_name)
  cartoon_image = combinator(quantized_image, inverted_edges, img_name + "quantized_" + str(quantization_level))

  quantization_level = 10
  quantized_image = quantizer(smoothed_image, quantization_level, img_name)
  cartoon_image = combinator(quantized_image, inverted_edges, img_name + "quantized_" + str(quantization_level))

  quantization_level = 12
  quantized_image = quantizer(smoothed_image, quantization_level, img_name)
  cartoon_image = combinator(quantized_image, inverted_edges, img_name + "quantized_" + str(quantization_level))

  quantization_level = 16
  quantized_image = quantizer(smoothed_image, quantization_level, img_name)
  cartoon_image = combinator(quantized_image, inverted_edges, img_name + "quantized_" + str(quantization_level))

  ######################################################################################################################

  img_name = "image1"
  smoothed_image = gaussian_filter(img1, (2, 2, 0))
  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 30}
  edge = DoG_filter_func(smoothed_image, parameters,  img_name + "Final")
  quantization_level = 4
  cartooner(img1,edge,smoothed_image, parameters, quantization_level, img_name + "Final")


  ######################################################################################################################

  img_name = "image2"
  img_2_path ='report/data/Persian+Cat+Facts+History+Personality+and+Care+_+ASPCA+Pet+Health+Insurance+_+white+Persian+cat+resting+on+a+brown+sofa-min.jpg'
  img2 = read_image_cv2(img_2_path)
  cv2.imwrite('report/result/'+img_name + "_orginal"+".jpg",img2)
  smoothed_image = gaussian_filter(img2, (2, 2, 0))
  parameters = {'sigma': 0.9, 'k':1.5, "threshold": 50} # 1.8, 1.1, 30
  edge = DoG_filter_func(smoothed_image, parameters,  img_name + "Final")
  quantization_level = 12
  cartooner(img2,edge,smoothed_image, parameters, quantization_level, img_name + "Final")

  ######################################################################################################################

  img_name = "image3"
  img_3_path = 'report/data/dd1f8bf8be534e33aee2e85983007e09.jpg'
  img3 = read_image_cv2(img_3_path)
  cv2.imwrite('report/result/'+img_name + "_orginal"+".jpg",img3)
  smoothed_image = gaussian_filter(img3, (2, 2, 0))
  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 30}
  edge = DoG_filter_func(smoothed_image, parameters,  img_name + "Final")
  quantization_level = 4
  cartooner(img3,edge,smoothed_image, parameters, quantization_level, img_name + "Final")

  ######################################################################################################################

  img_name = "image4"
  img_4_path = 'report/data/8682484.jpg'
  img4 = read_image_cv2(img_4_path)
  cv2.imwrite('report/result/'+img_name + "_orginal"+".jpg",img4)
  smoothed_image = gaussian_filter(img4, (2, 2, 0))
  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 30}
  edge = DoG_filter_func(smoothed_image, parameters,  img_name + "Final")
  quantization_level = 4
  cartooner(img4,edge,smoothed_image, parameters, quantization_level, img_name + "Final")

  ######################################################################################################################

  img_name = "image5"
  img_5_path = 'report/data/kapadokya-balon-turu-fiyatlari-nedir.jpg'
  img5 = read_image_cv2(img_5_path)
  cv2.imwrite('report/result/'+img_name + "_orginal"+".jpg",img5)
  smoothed_image = gaussian_filter(img5, (2, 2, 0))
  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 30}
  edge = DoG_filter_func(smoothed_image, parameters,  img_name + "Final")
  quantization_level = 4
  cartooner(img5,edge,smoothed_image, parameters, quantization_level, img_name + "Final")

  ######################################################################################################################

  img_name = "image6"
  img_6_path = 'report/data/Tower-Bridge.jpg'
  img6 = read_image_cv2(img_6_path)
  cv2.imwrite('report/result/'+img_name + "_orginal"+".jpg",img6)
  smoothed_image = gaussian_filter(img6, (2, 2, 0))
  parameters = {'sigma': 1.8, 'k':1.1, "threshold": 30}
  edge = DoG_filter_func(smoothed_image, parameters,  img_name + "Final")
  quantization_level = 4
  cartooner(img6,edge,smoothed_image, parameters, quantization_level, img_name + "Final")

  cv2.waitKey(500)
  cv2.destroyAllWindows()


if __name__ == "__main__":
   main()