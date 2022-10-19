#Nama   : Dwi Ramadhaniasari
#NIM    : 21091397057
#Kelas  : Manajemen Informatika 2021 A
#Multi neuron (Pakai numpy)

#membuat array dengan numpy
#Impor modul Numpy.
import numpy as np

# masukan variabel angka dengan jumlah input 10
inputs = [3.0, 8.0, 2.0, 9.0, 4.0, 1.0, 7.0, 5.0, 6.0, 10.0]

#Bobot diteruskan ke perception
#panjang bobot harus sesuai dengan panjang input ialah 10, lalu jumlah bobot sesuai dengan jumlah neuron yaitu 5
weights = [[0.6, 0.8, 0.4, 0.2, 0.3, 0.5, 0.7, 0.1, 0.0, -0.8],
           [1.0, 11.00, 0.22, 3.2, 1.2, 8.3, 5.4, 1.49, 5.79, 3.29],
           [0.64, 7.6, 3.7, 0.26, 8.24, -0.30, -0.38, 0.56, 0.34, -3.1],
           [1.89, 9.29, 0.24, 3.3, 6.2, 4.50, 9.23, 7.7, 0.35, 0.32],
           [9.0, 5.3, 0.11, 2.2, -0.63, -0.84, 7.7, -2.1, 0.41, -0.26]]

#bias untuk perception tertentu
#dengan jumlah bias berdasarkan jumlah neuron yaitu 5
biases = [9.1, 4.2, 1.0, 1.1, 0.6]

#Panggil fungsi np.dot
#Ambil hasil kali titik antara bobot dan masukan
#dan tambahkan bias ke nilai penjumlahan
layer_outputs = np.dot(weights, inputs) + biases

#cetak output
print(layer_outputs)