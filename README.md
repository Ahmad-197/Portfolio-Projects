# Electrical Impedance Tomography Data Augmentation Using Continuous Conditional Generative Adversarial Networks



Electrical Impedance Tomography (EIT) is a non-invasive imaging method that visualises the internal electrical conductivity of the body, or any unknown object. The EIT system collects voltage data from electrodes placed around an object by applying small electrical currents through a pair of them and recording the resulting boundary voltages on the remaining pairs. EIT involves a forward model, in which the internal electrical conductivity of the object is known, and the resulting voltages are found. Another important aspect of EIT is the inverse problem, which uses
the voltages obtained from the forward model to estimate the internal conductivity distribution.

EIT image reconstruction is a highly ill-posed, nonlinear inverse problem. Due to the benefits of machine learning (ML) in addressing ill-posed, nonlinear inverse problems such as EIT, there has
been an increase in research on data-driven reconstruction methods. However, several challenges must be addressed before ML-based EIT can be effectively applied, particularly the need for large, high-quality datasets, which are often difficult and timeconsuming to obtain. Data biases are also a critical factor that must be carefully considered.

To address the issue of limited data availability or data biases, several techniques can be used, with data augmentation being one of the most promising. Data augmentation remains an underexplored yet highly valuable approach for EIT applications. The use of generative adversarial networks (GANs) for data augmentation in EIT is an area that warrants further investigation. GANs have revolutionised generative modelling: the generator, G creates synthetic samples from random noise, while the discriminator, D attempts to distinguish between real data and the outputs
of G. This adversarial training process drives G to generate increasingly realistic data. A limitation of traditional GANs is the inability to control the type of data an unconditioned
generative model produces. This leads to the use of continuous conditional GANs (CcGANs). By employing CcGANs to synthesise high-fidelity, physiologically relevant EIT voltage data conditioned
on critical parameters such as aortic pressure, reliance on expensive and ethically complex animal experiments could be significantly reduced. We aim to generate EIT voltages, particularly at lower aortic pressures, as the real dataset we obtained contained considerably fewer data points in this range.

Before training the CcGAN model with experimental EIT data, we first trained it using simulated EIT voltage data conditioned on the angle of the anomaly. The model was rigorously trained
and fine-tuned by varying datasets and hyperparameters to achieve optimal performance. Using the simulated data, we were able to obtain a G capable of producing reliable results. However, the
model’s performance declined significantly when a substantial portion of the data was removed during training, and it was asked to generate the missing data.

The CcGAN trained on the real dataset learned a smooth, population-level mapping from aortic pressure to EIT voltage patterns, accurately reflecting the pressure conditioning without collapsing into discrete modes. The G effectively captured the primary influence of pressure changes on the voltage patterns, but it did not fully reproduce the range of variability observed in the real
measurements.

Thus, in its current form, the use of this CcGAN model cannot be reliably extended to generate experimental EIT data. Modifications, such as incorporating physics-informed learning frameworks into the CcGAN architecture, could be explored as part of future work.
