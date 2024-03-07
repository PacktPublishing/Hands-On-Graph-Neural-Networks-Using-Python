### [Packt Conference : Put Generative AI to work on Oct 11-13 (Virtual)](https://packt.link/JGIEY)

<b><p align='center'>[![Packt Conference](https://hub.packtpub.com/wp-content/uploads/2023/08/put-generative-ai-to-work-packt.png)](https://packt.link/JGIEY)</p></b> 
3 Days, 20+ AI Experts, 25+ Workshops and Power Talks 

Code: <b>USD75OFF</b>

# Hands-On Graph Neural Networks Using Python

<a href="https://www.packtpub.com/product/hands-on-graph-neural-networks-using-python/9781804617526?utm_source=github&utm_medium=repository&utm_campaign=9781804617526"><img src="https://static.packt-cdn.com/products/9781804617526/cover/smaller" alt="Hands-On Graph Neural Networks Using Python" height="256px" align="right"></a>

This is the code repository for [Hands-On Graph Neural Networks Using Python](https://www.packtpub.com/product/hands-on-graph-neural-networks-using-python/9781804617526?utm_source=github&utm_medium=repository&utm_campaign=9781804617526), published by Packt.

**Practical techniques and architectures for building powerful graph and deep learning apps with PyTorch**

## What is this book about?
Graph neural networks are a highly effective tool for analyzing data that can be represented as a graph, such as social networks, chemical compounds, or transportation networks. The past few years have seen an explosion in the use of graph neural networks, with their application ranging from natural language processing and computer vision to recommendation systems and drug discovery.

This book covers the following exciting features: 
* Understand the fundamental concepts of graph neural networks
* Implement graph neural networks using Python and PyTorch Geometric
* Classify nodes, graphs, and edges using millions of samples
* Predict and generate realistic graph topologies
* Combine heterogeneous sources to improve performance
* Forecast future events using topological information
* Apply graph neural networks to solve real-world problems

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1804617520) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
DG = nx.DiGraph()
DG.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'), ('C', 'G')])
```


**Following is what you need for this book:**
This book is for machine learning practitioners and data scientists interested in learning about graph neural networks and their applications, as well as students looking for a comprehensive reference on this rapidly growing field. Whether you’re new to graph neural networks or looking to take your knowledge to the next level, this book has something for you. Basic knowledge of machine learning and Python programming will help you get the most out of this book.	 

Basic knowledge of Python will help you get more from the examples.	
With the following software and hardware list you can run all code files present in the book (Chapter 2-14).

### Software and Hardware List

You should have a basic understanding of graph theory and machine learning concepts, such as
supervised and unsupervised learning, training, and the evaluation of models to maximize your
learning experience. Familiarity with deep learning frameworks, such as PyTorch, will also be useful,
although not essential, as the book will provide a comprehensive introduction to the mathematical
concepts and their implementation.

| Software required                      | OS required                        |
| ------------------------------------   | -----------------------------------|
| Python 3.8.15                          | Windows, Mac OS X, and Linux (Any) |                                                            
| PyTorch 1.13.1                         | Windows, Mac OS X, and Linux (Any) |
| PyTorch Geometric 2.2.0                | Windows, Mac OS X, and Linux (Any) |

To install Python 3.8.15, you can download the latest version from the official Python website: https://www.python.org/downloads/. We strongly recommend using a virtual environment, such as venv or conda. Optionally, if you want to use a Graphics Processing Unit (GPU) from NVIDIA to accelerate training and inference, you will need to install CUDA and cuDNN:

CUDA is a parallel computing platform and API developed by NVIDIA for general computing on GPUs. To install CUDA, you can follow the instructions on the NVIDIA website: https://developer.nvidia.com/cuda-downloads. 

cuDNN is a library developed by NVIDIA, which provides highly optimized GPU implementations of primitives for deep learning algorithms. To install cuDNN, you need to create an account on the NVIDIA website and download the library from the cuDNN download page: https://developer.nvidia.com/cudnn. 

You can check out the list of CUDA-enabled GPU products on the NVIDIA website: https://developer.nvidia.com/cuda-gpus. To install PyTorch 1.13.1, you can follow the instructions on the official PyTorch website: https://pytorch.org/. You can choose the installation method that is most appropriate for your system (including CUDA and cuDNN).

To install PyTorch Geometric 2.2.0, you can follow the instructions in the GitHub repository: https://pytorch-geometric.readthedocs.io/en/2.2.0/notes/installation.html.
You will need to have PyTorch installed on your system first.

Chapter 11 requires TensorFlow 2.4. To install it, you can follow the instructions on the official TensorFlow website: https://www.tensorflow.org/install. You can choose the installation method that is most appropriate for your system and the version of TensorFlow you want to use.

Chapter 14 requires an older version of PyTorch Geometric (version 2.0.4). It is recommended to create a specific virtual environment for this chapter.

Chapter 15, Chapter 16, and Chapter 17 require a high GPU memory usage. You can lower it by decreasing the size of the training set in the code.
Other Python libraries are required in some or most chapters. You can install them using pip install <name==version>, or using another installer depending on your configuration (such as conda).

Here is the complete list of required packages with the corresponding versions:
* torch==1.13.1+cu117
* torchvision==0.14.1+cu117
* torchaudio==0.13.1+cu117
* pandas==1.5.2
* gensim==4.3.0
* torch-scatter==2.1.0+pt113cu117
* torch-sparse==0.6.16+pt113cu117
* torch-cluster==1.6.0+pt113cu117
* torch-spline-conv==1.2.1+pt113cu117
* torch-geometric==2.2.0
* networkx==2.8.8
* matplotlib==3.6.3
* node2vec==0.4.6
* seaborn==0.12.2
* scikit-learn==1.2.0
* tensorflow-gpu~=2.4
* deepchem==2.7.1
* torch-geometric-temporal==0.54.0
* captum==0.6.0

The complete list of requirements is available on GitHub at https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python. 
Alternatively, you can directly import notebooks in Google Colab at https://colab.research.google.com.

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://packt.link/gaFU6).

## Errata

* Page 62: The number of ratings should be _**55375**_, not _**48,580**_.

* Page 109: Formula of GAT should be
 ![image](https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python/assets/108388790/8d00dd02-8296-4a85-9f3a-c1b91a386f4b)


### Related products <Other books you may enjoy>
* Network Science with Python [[Packt]](https://www.packtpub.com/product/network-science-with-python/9781801073691) [[Amazon]](https://www.amazon.com/dp/B0BJKP7R4P)

* 3D Deep Learning with Python [[Packt]](https://www.packtpub.com/product/3d-deep-learning-with-python/9781803247823) [[Amazon]](https://www.amazon.com/dp/B0BJVQG1VS)

## Get to Know the Author
**Maxime Labonne**
is a senior applied researcher at J.P. Morgan with a Ph.D. in machine learning
and cyber security from the Polytechnic Institute of Paris. During his Ph.D., Maxime worked on
developing machine learning algorithms for anomaly detection in computer networks. He then joined
the AI Connectivity Lab at Airbus, where he applied his expertise in machine learning to improve the
security and performance of computer networks. He then joined J.P. Morgan, where he now develops
techniques for solving a variety of challenging problems in finance and other domains. In addition
to his research work, Maxime is passionate about sharing his knowledge and experience with others
through Twitter (@maximelabonne) and his personal blog.

### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781804617526">https://packt.link/free-ebook/9781804617526 </a> </p>
