

#---pyspark implementation of Kernel NMF --------------
#-----------------spark 2.1.0 --------------------------

print('starting KOGNMF RS script')
import findspark
findspark.init()
import pyspark

from pyspark import SparkConf
from pyspark import SparkContext

#configure for cluster ....euler or local
#conf = SparkConf()
#conf.setMaster('spark://url:7077')
#conf.setAppName('spark-basic')
#sc = SparkContext(conf=conf)

sc = pyspark.SparkContext(master='local[*]')
print(sc)


from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix

#local matrix
#dm2 = Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])

#distributed matrix
# Row matrix, IndexedRowMatrix,  CoordinateMatrix, BlockMatrix

from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix

#import from file system ....
# Xrdd = sc.textFile("data.txt")
Xrdd = sc.parallelize([IndexedRow(0, [1, 1, 0]), IndexedRow(1, [1, 0, 0]),
                              IndexedRow(2, [1, 1, 0]), IndexedRow(3, [0, 0, 1])])

nNumberPerBlock=3
mNumberPerBlock=2
kNumberPerBlock=3

print(Xrdd)
Xrm = IndexedRowMatrix(Xrdd)
Xbm = Xrm.toBlockMatrix(mNumberPerBlock, nNumberPerBlock)
print(Xbm.toLocalMatrix())
Xbm.validate()

import numpy as np

#matrix H : kxn
k = 2;
n = Xbm.numCols()
m = Xbm.numRows()

Hindexedrows = []
for i in range(0,k):
	Hindexedrows.append((i, np.random.random_sample((n,))))

H_bm = IndexedRowMatrix(sc.parallelize(Hindexedrows)).toBlockMatrix(kNumberPerBlock, nNumberPerBlock)
H_bm.validate()
H_local = H_bm.toLocalMatrix()

print(H_local)

#matrix F : nxk
Findexedrows = []
for i in range(0,n):
	Findexedrows.append((i, np.random.random_sample((k,))))

F_bm = IndexedRowMatrix(sc.parallelize(Findexedrows)).toBlockMatrix(nNumberPerBlock, kNumberPerBlock)
F_bm.validate()

print(F_bm.toLocalMatrix())

K = Xbm.transpose().multiply(Xbm)
K.validate()
print(K.toLocalMatrix())


alpha = 10
mu = 100

from pyspark.mllib.linalg import Matrix

#X_bm : block matrix distributed rdd
#X_np : numpy local matrix
#X_local: local matrix MLlib structure

num_iter = 100
for i in range(0,num_iter):
	#update for H = H.* A./B
	FK_bm = F_bm.transpose().multiply(K)
	#block matrix rdd.toLocalMatrix() -> local rdd matrix.toArray() -> numpy matrix
	A_np = np.multiply( alpha * FK_bm.toLocalMatrix().toArray()  ,2*mu*H_local.toArray())
	FKFH_bm = FK_bm.multiply(F_bm).multiply(H_bm)
	HHH_np = np.matmul(np.matmul(H_local.toArray(), H_local.toArray().transpose()), H_local.toArray())
	B_np = np.multiply( alpha*FKFH_bm.toLocalMatrix().toArray(), 2*mu*HHH_np)
	#convert numpy to dense rdd :
	#Hlocal = Matrices.dense(k,n,np.multiply( np.divide(A_np, B_np), Hlocal.toArray() ).flatten('F'))
	H_np = np.multiply( np.divide(A_np, B_np), H_local.toArray())
	H_bm = IndexedRowMatrix(sc.parallelize( list(enumerate(H_np.tolist()) ))).toBlockMatrix(kNumberPerBlock, nNumberPerBlock)
	H_local = H_bm.toLocalMatrix()
	print(H_local)
	#update for F
	KH_bm = K.multiply( H_bm.transpose() )
	KFHH_bm = K.multiply(F_bm).multiply(H_bm).multiply(H_bm.transpose())
	F_np = np.multiply( F_bm.toLocalMatrix().toArray() , np.divide( KH_bm.toLocalMatrix().toArray(), KFHH_bm.toLocalMatrix().toArray() ))
	F_bm = IndexedRowMatrix(sc.parallelize( list(enumerate(F_np.tolist()) ))).toBlockMatrix(nNumberPerBlock, kNumberPerBlock)
	print(F_bm.toLocalMatrix())
print(H_local)
#recommendations Y= X*F*H
Y = np.matmul(Xbm.toLocalMatrix().toArray() , np.matmul(F_np, H_np))
print(Y)

#store results ....



