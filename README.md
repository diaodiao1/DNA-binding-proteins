# MsDBP
Exploring DNA-binding Proteins by Integrating Multi-scale Sequence Information with Deep Neural Network
------

Dependency:</br>
Python 3.6.2</br>
Numpy 1.13.1</br>
Scikit-learn 0.19.0</br>
Tensorflow 1.3.0</br>
keras 2.0.8</br></br>


Usage:</br>
Run this file from command line.</br>
For example:</br>

python MsDBN_predict.py -dataset_P=PDB2272_P  -dataset_N=PDB2272_N</br>
python MsDBN_predict.py -dataset_P=DBP2858</br>
python MsDBN_predict.py -dataset_N=NDBP3723</br>
where PDB2272_P, PDB2272_N, DBP2858, NDBP3723 are the datasets used in our study.

python MsDBN_predict.py -dataset_P=DBP_fasta_file  -dataset_N=NDBP_fasta_file</br>
it can be applied to other datesets.

python MsDBN_predict.py -dataset_P=DBP_fasta_file</br>
it can be used to find out new DBPs.

python MsDBN_predict.py -dataset_N=NDBP_fasta_file</br>
it can be used to FPR of the predictor.</br></br>


fasta format:</br>
&gt;P27204|1</br>
AKKRSRSRKRSASRKRSRSRKRSASKKSSKKHVRKALAAGMKNHLLAHPKGSNNFILAKKKAPRRRRRVAKKVKKAPPKARRRVRVAKSRRSRTARSRRRR</br>
&gt;Q58183|0</br>
MERLPYEIVSTIFRKAILHYVLIRGTTYPQSLAENLNISKGLASSFLRLCSALNIMKRERAGHKVLYSFTSKGLAILKRLAPEIFDLSFSSVFEQLPKKKIATKYYPVDKIGFEISWKEDKLGGIVFSFFDSNGEHLGDVFRSNKGYWWCVICQSDTCKHIDYLKRLYKTLKNQD</br></br>


Contact us:</br>
dxqllp@163.com</br>
