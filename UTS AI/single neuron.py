#Nama   : Dwi Ramadhaniasari
#NIM    : 21091397057
#Kelas  : Manajemen Informatika 2021 A
#single neuron (Pakai numpy)

#Contoh produk titik menggunakan numpy
#Impor modul Numpy.
import numpy as np

#masukan perception dengan jumlah input 10
inputs = [12, 11, 1, 9, 3, 2, 7, 8, 6, 11]

#Bobot diteruskan ke perception
#panjang bobot harus sesuai dengan panjang input yaitu 10, lalu jumlah bobot sesuai dengan jumlah neuron yaitu 1
weights = [0.6, 0.8, 0.1, 0.3, 0.9, 0.7, 0.4, 0.2, 0.5, -0.11]

#bias untuk perception tertentu
#dengan jumlah bias berdasarkan jumlah neuron
bias = 9

#Ambil hasil kali titik antara bobot dan masukan
#dan tambahkan bias ke nilai penjumlahan
output = np.dot(weights, inputs) + bias

#mencetak output
print(output)