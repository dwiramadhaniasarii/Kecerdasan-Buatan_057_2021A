#Nama   : Dwi Ramadhaniasari
#NIM    : 21091397057
#Kelas  : Manajemen Informatika 2021 A
#Multi Neuron Batch Input

#Impor modul Numpy.
import numpy as np

#memasukan variabel dengan matriks 6x10 (input 10 dan batch 6)
inputs = [[0.0, 2.4, 0.2, 3.0, 0.5, 1.0, 0.7, 9.0, 0.4, 2.0],
          [2.2, 1.1, 3.4, 0.9, 6.0, 5.0, 6.0, 9.0, 2.7, 2.9],
          [1.5, 3.2, 4.0, 1.0, 3.4, 5.9, 9.3, 7.7, 1.5, 4.8],
          [7.3, 6.7, 7.8, 1.9, 8.8, 2.1, 7.0, 4.4, 8.8, 3.8],
          [7.2, 9.9, 3.9, 1.6, 1.5, 1.4, 8.7, 6.6, 2.6, 1.3],
          [0.3, 4.3, 0.7, 6.6, 8.1, 1.7, 8.0, 3.0, 5.0, 4.0]]

#panjang weights sinkron dengan panjang input yaitu 10, lalu jumlah weights sama dengan jumlah neuron ialah 5
weights = [[1.1, 3.1, 2.3, 2.2, 4.2, 5.9, 4.3, 8.7, 2.1, 9.7],
           [4.0, 8.1, 1.3, 7.0, 4.3, 2.1, 3.0, 2.0, 4.7, 4.9],
           [9.2, 6.5, 8.4, 7.5, 1.8, 3.7, 4.2, 2.5, 2.9, 1.8],
           [2.9, 1.1, 0.2, 1.8, 2.4, 3.0, 1.0, 4.0, 8.8, 1.6],
           [6.6, 1.7, 2.7, 3.7, 8.3, 4.9, 7.7, 5.2, 9.8, 3.0]]

#total bias sinkron dengan total neuron ialah 5
biases = [1.0, 3.1, 5.5, 2.8, 7.9]

#Panggil fungsi np.dot
#masukan input
#panggil fungsi array
#kalikan bobot dengan Transpos matriks
#tambahakan bias
layer_outputs = np.dot(inputs, np.array(weights).T) + biases

#cetak ouputs
print(layer_outputs)