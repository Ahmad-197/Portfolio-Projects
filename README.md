# **EIT Data Acquisition and Evaluation and Reconstruction with Data-Driven Methods**



## Table of Contents


1. [Introduction](#1-introduction)
2. [EIT Simulation](#2-eit-simulation)
3. [Experimental Setup](#3-experimental-setup)
5. [Variational Autoencoder](#4-variational-autoencoder)
6. [Mapper Models](#5-mapper-models)
   
   5.1. [Simple Mapper](#51-simple-mapper)

   5.2. [Sequential Mapper](#52-sequential-mapper)

   5.3. [LSTM Mapper](#53-lstm-mapper) 

7. [Results](#6-results)
8. [Citations](#7-citations)




## **1. Introduction**

Electrical impedance tomography (EIT) is a non-invasive imaging technique that was introduced by Barber and Brown in the early 1980s <a href="#ref2">[2]</a>. It involves injecting low alternating electric currents through surface electrodes into an object and measuring the resulting voltages on the remaining electrodes. Different conductivity distributions within the object influence the current flow and resultantly the measured voltages. The cross-sectional conductivity distribution can be reconstructed from the measured voltages by solving the ill-pose inverse EIT problem. For instance, in respiratory monitoring, changes in conductivity caused by airflow in the lungs are visualized [<a href="#ref3">3</a>, <a href="#ref1">1</a>]. Due to the benefits of machine learning (ML) in addressing ill-posed, non-linear inverse problems such as EIT, there has been an increase in research on data-driven reconstruction methods. To create a successful generalized ML reconstruction model, a significant amount of high-quality training data is required. Therefore, an experimental environment should be designed, where the true conductivity distribution is known for a detailed and quantitative evaluation of the ML
model performance.

To enhance the reliability and quality of EIT imaging, the application of deep learning (DL) methods for image reconstruction has significantly increased in recent years, cf. [<a href="#ref4">4</a>, <a href="#ref6">6</a>, <a href="#ref5">5</a>]. Using a Long short-term memory (LSTM) in the reconstruction network can enhance robustness due to the periodicity of the breathing cycle.

## **2. EIT Simulation**

We used EIT simulation to make a "proof of concept" for our reconstruction approach and generate the EIT dataset that is used to train our VAE. The simulation could be broken down into two main parts; modelling anomaly and solving the forward problem. The anomaly is modelled based on the equation below:

$anomaly = 0.075 \times \left(1 + \sin\left(\frac{2 \pi \times \text{Total Time}}{\text{Time Period}}\right)\right) + 0.2$

The radius of the anomaly is approximated as a sine wave that ranges from 0.2 to 0.35. This approximation is based on the changes in the balloon's radius as in the [experimental setup](#3-experimental-setup). The anomaly is a balloon in the experimental setup which is inflated and deflated similar to the inflation and deflation of lungs. So, the breathing cycle of an adult human is mimicked. Since, with each inhalation and exhalation cycle the volume of the lungs increases and decreases respectively. Similarly, the size of the anomaly would also change accordingly. Hence, a sinusoidal wave is used to mimic such changes. As the cross-sectional view of the balloon is circular, so anomaly is also treated as such.

The region of interest where the anomaly has to be detected is modelled as a 2D mesh with triangular mesh elements as shown in the Fig. 1.

<p align="center">
  <img src="https://github.com/user-attachments/assets/cc086c60-feb8-4f5b-bb47-a4b4e0918bbc" alt="Empty_mesh">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 1: Modelled mesh for permittivity distribution</em>
</p>

This is done so that it becomes easier to numerically solve the forward problem. A mesh is modelled at each sampling instant. Each mesh element has its permittivity value. 
We then introduce an anomaly to each of the meshes. Depending on the size of the anomaly the permittivity for the proportional number of mesh elements in a mesh is set to be 10 and the permittivity for the rest of the elements is set at 1. The figure below shows the anomaly plot and how its size varies in the mesh.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4de10343-0b8b-4951-9308-f39af70d220d" alt="Mesh_with_anomaly_animation">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 2: Variation in anomaly size over a single time period</em>
</p>

The permittivity associated with each of these mesh elements is stored in a single array. Hence, by simulating an anomaly we get a known permittivity distribution which is then used to find out voltages. 

A system with thirty-two electrodes is simulated with an opposite injection pattern with a skip of sixteen electrodes. The injecting electrodes are the electrodes to which a sinusoidal electric current is injected and afterwards, the voltages at each of the remaining adjacent pairs are recorded. This pattern is repeated until all the possible injecting electrode patterns are simulated. In simulating EIT data we know the permittivity distribution and we use it to obtain the voltages at electrodes. Using permittivity to obtain these voltages is also known as the EIT forward problem. 


## **3. Experimental Setup**

For the experimental setup, a balloon was submerged in a tank filled with the saline solution which was made by mixing distilled water and salt so that the resulting solution had about 9 ppt salt concentration. The volume of saline was approximately 15 litres. The balloon was inflated and deflated constantly using a medical device called “SALVIA”. This device could also be used to provide oxygens to humans so by using this device the breathing cycle in humans was mimicked and the balloon in the tank is analogous to the lungs in the body. As the balloon inflates and deflates the level of the saline rises and falls accordingly. The change in level is measured by an ultrasonic sensor which was fixed at the top of the tank as shown in Fig. 3. The ultrasonic sensor is connected to an Arduino, which runs a script that allows to record the saline level as measured by the ultrasonic sensor.

An array of thirty-two electrodes was connected and circled the tank to record the boundary voltages. These electrodes were connected to a Scio Spec device which injected a small sinusoidal  electric current into two electrodes and then measured the potential differences at all the adjacent electrode pairs including the ones that were used as injecting electrodes. This is repeated until all the possible combinations are achieved and in the end, we get a 1024-size array. The voltages recorded were complex i.e. the potential difference was AC in nature. The figure below shows the experimental setup used.

<table align="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/cf44ba70-8008-47d7-959f-8fa03ddc7be5" alt="exp_setup1 (4)"></td>
    <td><img src="https://github.com/user-attachments/assets/6c33351d-2226-4a0e-948e-86a5fc6b55ac" alt="exp_setup2 (2)"></td></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><em>Fig. 3: Experimental setup side and top views</em></td>
  </tr>
</table>



With the change in the size of the balloon, the cross-sectional conductivity distribution changes and as a result, the boundary voltages were also changed. So, we recorded both the change in the saline level and voltages.  The raw data from the ultrasonic sensor was noisy and had fluctuations. So a median filter was applied to remove much of the noise and then to further improve this data Fast Fourier Transform (FFT) was applied to the data from the median filter. We used FFT to get the signal frequency components and took only a limited number of those signal frequency components (terms) from the spectrum to create a Fourier series representation of the signal. Then this Fourier series representation was reconstructed (to get a time-domain signal) using inverse FFT. This helped in further removing unwanted frequencies and smoothening of the signal. A comparison between these three signals is shown in the figure below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/75e414cc-024d-49cb-a8d0-99f6246956c9" alt="Filtered_Data">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 4: Raw and filtered data plots</em>
</p>

Since a balloon is an anomaly which is varied in size and which has to be detected later on using Machine Learning Methods. We need to find out the changes in the size of the balloon. For this we used our FFT data and inverted and scaled it between 0.20 to 0.35. The inversion was done because the ultrasonic sensor was measuring the distance between the top of the tank and the saline level.  As the balloon was inflated its size and resultantly the water level would also go up. So, the distance between the saline surface and the top of the tank would decrease. The figure below shows the balloon radius signal.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4a398f94-47ca-47e8-b44f-07a008232028" alt="balloon_radius">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 5: Balloon radius plot</em>
</p>

This is how we can find out the change in the size of the anomaly. As in simulation, the cross-section of the tank is modelled as a mesh, where the information related to the changes to the size of the anomaly is obtained as discussed above. As done in the EIT simulation section, the permittivity array for each of the anomaly samples is recorded. 

So, we get both permittivity and voltage information, which is then used for training neural networks.

### Augmented Dataset

In addition to the simulated and experimental datasets, a third dataset was also used to check the output of mappers. This is also called as 2nd simulation dataset in this report. Here, the anomaly was modelled using the same method as in the experimental setup. However, only a small dataset of saline level changes was collected. Later on, data augmentation was performed to get a larger dataset. Before data augmentation a clean subset of 8 time periods of the signal was selected from the collected data to which a median filter was applied and then we obtained the Fourier series representation as mentioned above. After scaling and inverting the filtered data, they were divided into 8 different chunks. These chunks were then shuffled randomly and added to the original dataset until the required number of samples were obtained.

The plot below shows the change in the radius of the anomaly.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a7ba7743-3d5c-4173-91e0-cb0003d4b547" alt="3rd_data_set_anomaly">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 6: Change of anomaly radius for augmented data</em>
</p>



Noise is also added to the mesh elements so that the signal-to-noise ratio (SNR) is 30. Note that voltages were obtained in the same way as in the simulation case i.e. by solving the forward problem. 



## **4. Variational Autoencoder**

Variational Autoencoder (VAE) is a type of autoencoder consisting of an encoder and a decoder that is used for unsupervised learning, it involves learning a model distribution from the input data such that the learned distribution is just like the distribution of the input data <a href="#ref7">[7]</a>. The encoder part of the VAE helps to learn a Gaussian distribution from the input data by producing the mean and standard deviation of the distribution from which a low-dimensional latent variable is sampled <a href="#ref7">[7]</a>. Thus, by using the encoder we can transform our input data from a higher dimension into a lower dimensional latent space. Afterwards, the decoder learns to transform this low-dimensional latent representation back to the original input data shape <a href="#ref7">[7]</a>. The figure below gives a high-level overview of the architecture of the VAE.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ce395b5-7980-49ce-af45-2f438d3ad3f9" alt="VAE_Architecture">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 7: VAE model</em>
</p>


We used permittivity data to train our VAE. We used the encoder to create a set of latent representations based on the permittivity data and then trained the decoder to learn to decode that latent representation to its original form. The figure below compares the original and reconstructed permittivity distribution using VAE.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d8685d5a-0b7a-4589-8487-1e9b468eb1c2" alt="VAE (1)">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 8: True vs predicted mesh plots using VAE</em>
</p>

Additionally, the figure below shows the box plot which represents the deviations in mesh elements. This helps to analytically evaluate the performance of the VAE. The mean deviation is 0.26 and the standard deviation is 0.439 denoting VAE is performing well.

<p align="center">
  <img  height=500 src="https://github.com/user-attachments/assets/96e018dc-19fb-46ac-a8d7-97ebec34533c" alt="VAE Boxplot(snr_30) (2)">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 9: Box plot of VAE for performance</em>
</p>


The trained VAE would help us in training our neural networks where we would use the latent representation to train the network and then the predicted latent representations would be passed through the decoder to reconstruct permittivity distribution. In short by using VAE we "generate a low dimensional manifold of approximate solutions, which allows conversion of the ill-posed EIT problem to a well-posed one" <a href="#ref7">[8]</a>. This is discussed in the next section.




## **5. Mapper Models**

After training VAE we would use it to train our neural networks (also called mappers). We would use the encoder to get latent representation using permittivity distribution and then train the mapper to map the voltages to that latent space. The mapper learns to map voltages to the latent space and then we use a decoder from the VAE to construct permittivity distribution from the voltage data. We used three different mapper models for training with simulation and experimental data. For each of the mapper same architecture has been used for training simulated and experimental EIT data. The only difference is between the input dimensions passed to the mapper due to different sizes of voltages array produced by simulation and experimental data.

### 5.1. Simple Mapper

A simple convolutional neural network is used having two 2D convolutional, one flatten and three dense layers. This model transforms an input of high dimensions into an output of eight features, which corresponds to the latent space.  This is exactly what is required as our decoder from VAE expects 8 dimensions of input. The Fig. 10 shows the data flow from the mapper to the VAE decoder.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/9b1bb36a-e1da-4ff8-8429-a726a42b0afd" alt="Simple_Mapper_Architecture">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 10: Simple mapper with decoder model</em>
</p>


The neural network learns to map the boundary voltages at a single time instant to the latent space created by the encoder of the VAE using permittivity distribution. 

The following figure represents the reconstruction of permittivity distribution using simulated data.


<p align="center">
  <img src="https://github.com/user-attachments/assets/17750a43-7709-4ea2-8c28-bd05f2c9abe7" alt="Simple_Mapper">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 11: Simple mapper true vs predicted mesh plots with simulated data</em>
</p>


The figure below shows the comparison between the actual and predicted permittivity distributions when experimental data was used. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2047acc4-9aab-4c2a-8bfc-d97954bb1755" alt="Simple_Mapper">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 12: Simple mapper true vs predicted mesh plots with experimental data</em>
</p>


### 5.2. Sequential Mapper

The sequential mapper is similar to the simple mapper, however, instead of passing only one EIT measurement we feed a time series of EIT measurements of length four into our mapper. Hence the name is sequential. In this case, we heuristically chose a sequence length of four. Using time series of EIT measurements allows the mapper to learn temporal dependencies between the data and make better predictions. The figure below shows the transformation of the input data from this mapper model and the decoder to the required output.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c007ca57-3318-4329-8e75-bb02b1015db1" alt="Seq_LSTM_Architecture">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 13: Sequential and LSTM mappers with decoder model</em>
</p>



We investigate in the [results](#6-results) section how does this change affect the performance of the mapper. The mapper consists of two 2D convolutional, two flatten and five dense layers.

The Fig. 14 shows the performance of this mapper when simulated data is used.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e53500c-69d2-4d8e-80b4-614365c01053" alt="Seq_Mapper">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 14: Sequential mapper true vs predicted mesh plots with simulated data</em>
</p>


The performance of the mapper with experimental data is displayed below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6de0e6cb-a5a3-46d4-acea-eb70e91ad3b1" alt="Seq_Mapper">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 15: Sequential mapper true vs predicted mesh plots with experimental data</em>
</p>


### 5.3. LSTM Mapper
LSTM mapper is very useful in learning long-term dependencies in data which with other recurrent neural networks (RNNs) suffer from gradient vanishing or exploding problem <a href="#ref8">[9]</a>. Thus, LSTM can learn temporal dependencies more accurately. The LSTM mapper used consists of two 2D convolutional, one flatten, three dense and two LSTM layers. The Fig. 13 shows the block diagram for the LSTM mapper and decoder and changes to the input data shape as it flows. A time series EIT measurement of length four is passed into this mapper as well and it is trained to predict the latent representation from the voltages. The results of the LSTM mapper are discussed in the next section. 

Mesh plots for true and predicted permittivity distribution using simulated data are shown in the figure below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a7619fc3-f1b0-406a-aa3d-0deed74009fb" alt="LSTM_Mapper">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 16: LSTM mapper true vs predicted mesh plots with simulated data</em>
</p>

The performance of LSTM with experimental data is as follows.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ac7932c3-a547-472b-a6a8-a2b8b6fdc845" alt="LSTM_Mapper">
</p>
<p align="center" style="font-size: smaller;">
  <em>Fig. 17: LSTM mapper true vs predicted mesh plots with experimental data</em>
</p>



It can be observed that in all three mappers, the reconstruction of simulated data was much better than the experimental data. This is due to the inherent noise and fluctuations in the experimental data which makes it difficult to make true predictions. The analytical performance of the mappers is discussed next. 


## **6. Results**

The Fig. 18 shows the box plots which represent the permittivity deviation in mesh elements for three of the mappers when simulated data was used. The simple mapper performed the worst with mean and standard deviations of 0.706 and 2.492 respectively. The performance of the sequential and LSTM mappers is comparable. LSTM performs better in terms of giving precise predictions as its permittivity error standard deviation is 1.797 and for sequential mapper this standard deviation is 2.433. However, the mean deviation for LSTM is -0.775 and 0.42 for sequential mapper. The predictions of LSTM are generally lower than the true value while they are higher for the sequential mapper. Overall, the sequential mapper gives relatively more accurate predictions on occasions but there is high variability in its predictions. The performance of LSTM is comparatively more consistent.

<table align="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a6e309ab-9ae1-42a5-8e83-a4f4fd316b64" alt="Simple_Mapper_Boxplot(snr_30)"></td>
    <td><img src="https://github.com/user-attachments/assets/285e4e4e-93d1-4aa9-85b8-8c17c347e5cf" alt="Seq_Mapper_Boxplot(snr_30)"></td>
    <td><img src="https://github.com/user-attachments/assets/fdc51fe3-f38e-442b-80c1-b1949bae7c58" alt="LSTM_Mapper_Boxplot(snr_30)"></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><em>Fig. 18: Box plot comparison of all the mappers with simulated data</em></td>
  </tr>
</table>


The performance of the simple mapper becomes worse as the complexity of the data is increased. This is evident from its performance with the experimental data as shown in Fig. 19. With experimental data the difference between the performance of the simple and other mappers becomes way larger. Here the standard deviation of permittivity error is 48.153 and the mean deviation is 12.686 suggesting that it struggles to learn complex patterns. With experimental data both sequential and LSTM mappers underestimate their predictions as they have negative mean deviations of -2.964 and -2.884 respectively. However, in this case, the predictions of the LSTM mapper are slightly more accurate. But in terms of variability in predictions; the LSTM mapper has more variations. The standard deviation is 18.53 and 15.82 for LSTM and sequential mappers respectively. The sequential mapper predicts consistently lower permittivity in comparison with the LSTM mapper as its median deviation is -2 and for the LSTM mapper, it is 0. 


<table align="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/391ee182-689b-493e-b0bc-511b0b8fdbb4" alt="Simple_Mapper_Boxplot"></td>
    <td><img src="https://github.com/user-attachments/assets/eb35b584-f1ed-44aa-bb69-5cf7366a3d3c" alt="Seq_Mapper_Boxplot(snr_30)"></td>
    <td><img src="https://github.com/user-attachments/assets/0341754e-dbca-4d60-8f7a-3802f3423e1e" alt="LSTM_Mapper_Boxplot(snr_30)"></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><em>Fig. 19: Box plot comparison of all the mappers with experimental data</em></td>
  </tr>
</table>


There is a slight performance difference in simulation and experimental data between LSTM and sequential mappers. However, the performance of LSTM is more consistent with regards to it being underestimating the values of its predictions. The sequential mapper displayed variance in this regard with overestimating permittivity in the case of simulation case and underestimating in the other case. So, the behaviour of the LSTM mapper is more consistent with respect to its predicted permittivity estimations. Moreover, in comparing the performance of the mappers more weightage is given to their performances in the experimental scenario as it is more closer to reality. Since the predictions of the LSTM mapper are more accurate in this case and it exhibits the same type of behaviour in both of the cases it is more reliable to be used. Furthermore, the percentage difference in the standard deviation is 35.4% in the simulation case while in experimental it is 17.1%. Thus, further solidifying the case for the usage of LSTM.  

In addition to testing out the mapper performances with these two datasets, a third (augmented) dataset was also used to further compare the performances. The results for the third dataset are visualized below. 


<table align="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/3ac06e36-692c-4985-943c-649f4153851d" alt="Simple_Mapper_Boxplot"></td>
    <td><img src="https://github.com/user-attachments/assets/0ed2a431-5ff5-4fbb-92e0-638152f017f6" alt="Seq_Mapper_Boxplot(snr_30)"></td>
    <td><img src="https://github.com/user-attachments/assets/2667c53f-b072-4901-8ada-954b3ab02cfb" alt="LSTM_Mapper_Boxplot(snr_30)"></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><em>Fig. 20: Box plot comparison of all the mappers with augmented data</em></td>
  </tr>
</table>

In this case as well the simple mapper's performance is the worst among the three with standard and mean deviations of 4.015 and 0.828 respectively. LSTM mapper performed significantly better than both of the other mappers with mean and standard deviations of -0.097 and 2.178 respectively. While for sequential mapper these values were -0.834 and 2.549. 

After comparing the performance of the three mappers with three different types of datasets, the LSTM mapper has stood out to be the best among the three followed by the sequential mapper. LSTM mapper has displayed consistent and relatively accurate predictions.






## **7. Citations**

<a id="ref1"></a> [1] Liliana Borcea. ”Electrical impedance tomography“. In: Inverse Problems 18.6 (2002), R99. DOI: 10.1088/0266-5611/18/6/201. URL: [https://dx.doi.org/10.1088/02665611/18/6/201 ](https://iopscience.iop.org/article/10.1088/0266-5611/18/6/201) 

<a id="ref2"></a> [2] BH Brown, DC Barber und AD Seagar. ”Applied potential tomography: possible clinical applications“. In: Clinical Physics and Physiological Measurement 6.2 (1985), S. 109

<a id="ref3"></a> [3] B.M. Eyuboglu, B.H. Brown und D.C. Barber. ” In vivo imaging of cardiac related impedance changes“. In: IEEE Engineering in Medicine and Biology Magazine 8.1 (1989), S. 39 45. DOI: 10.1109/51.32404 

<a id="ref4"></a> [4] Xiuyan Li u.a. ”An image reconstruction framework based on deep neural network for electrical impedance tomography“. In: IEEE International Conference on Image Processing (ICIP). 2017, S. 3585–3589. DOI: 10.1109/ICIP.2017.8296950 

<a id="ref5"></a> [5] Hao Yu u.a. ”High-resolution conductivity reconstruction by electrical impedance tomography using structure-aware hybrid-fusion learning“. In: Computer Methods and Programs in Biomedicine 243 (2024), S. 107861. issn: 0169-2607. doi: https://doi.org/10.1016/j.cmpb.2023.107861. URL: https://www.sciencedirect.com/science/article/pii/S0169260723005278

<a id="ref6"></a> [6] Liying Zhu u.a. ”Electrical Impedance Tomography Guided by Digital Twins and Deep Learning for Lung Monitoring“. In: IEEE Transactions on Instrumentation and Measurement 72 (2023), S. 1–9. DOI: 10.1109/TIM.2023.3298389 

<a id="ref7"></a> [7] C. Doersch, "Tutorial on Variational Autoencoders," arXiv, Aug. 2016, DOI: 10.48550/arXiv.1606.05908. [Online]. url: https://doi.org/10.48550/arXiv.1606.05908 

<a id="ref8"></a> [8] Seo, J. K., Kim, K. C., Jargal, A., Lee, K., & Harrach, B. (2019). A learning-based method for solving ill-posed nonlinear inverse problems: A simulation study of lung EIT. SIAM journal on Imaging Sciences, 12(3), 1275-1295.

<a id="ref8"></a> [9] Hochreiter, S., Schmidhuber, J.: Long short-term memory. Neural computation 9, 1735–80 (1997)










