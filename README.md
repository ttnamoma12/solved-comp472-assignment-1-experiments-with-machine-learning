Download Link: https://assignmentchef.com/product/solved-comp472-assignment-1-experiments-with-machine-learning
<br>
<h1>1        Experiments with Machine Learning</h1>

<strong>1.1 </strong>scikit-learn

For this assignment, you will use the scikit-learn machine learning framework to experiment with different machine learning algorithms and different data sets. The focus of this assignment lies more on the experimentations and analysis than on the implementation.

scikit-learn is an open-source machine learning library for Python (see <a href="http://scikit-learn.org/stable/">http://scikit-learn.org/stable/</a><a href="http://scikit-learn.org/stable/">)</a>, which provides an interface to program with a variety of different algorithms and built-in datasets. There are plenty of online documentation and examples of code online.

<h2>1.2      Data Sets</h2>

You must use the 2 datasets provided on Moodle (see the zip file DataSet-Release1). Both datasets are about the classification of black &amp; white images of size 32 × 32 that represent a character. For example, the image  represents the character ‘A’.

<strong>Dataset 1 </strong>contains images of the 26 uppercase letters [A – Z].

<strong>Dataset 2 </strong>contains images of 10 Greek letters.

Each character in the datasets is represented by an index, as indicated in the table below:

<table width="430">

 <tbody>

  <tr>

   <td width="58"> </td>

   <td colspan="4" width="195">Dataset 1</td>

   <td width="39"> </td>

   <td colspan="2" width="138">Dataset 2</td>

  </tr>

  <tr>

   <td width="58">index</td>

   <td width="39">char.</td>

   <td width="58">index</td>

   <td width="39">char.</td>

   <td width="58">index</td>

   <td width="39">char.</td>

   <td width="58">index</td>

   <td width="79">char.</td>

  </tr>

  <tr>

   <td width="58">0</td>

   <td width="39">A</td>

   <td width="58">10</td>

   <td width="39">K</td>

   <td width="58">20</td>

   <td width="39">U</td>

   <td width="58">0</td>

   <td width="79"><em>π </em>(pi)</td>

  </tr>

  <tr>

   <td width="58">1</td>

   <td width="39">B</td>

   <td width="58">11</td>

   <td width="39">L</td>

   <td width="58">21</td>

   <td width="39">V</td>

   <td width="58">1</td>

   <td width="79"><em>α </em>(alpha)</td>

  </tr>

  <tr>

   <td width="58">2</td>

   <td width="39">C</td>

   <td width="58">12</td>

   <td width="39">M</td>

   <td width="58">22</td>

   <td width="39">W</td>

   <td width="58">2</td>

   <td width="79"><em>β </em>(beta)</td>

  </tr>

  <tr>

   <td width="58">3</td>

   <td width="39">D</td>

   <td width="58">13</td>

   <td width="39">N</td>

   <td width="58">23</td>

   <td width="39">X</td>

   <td width="58">3</td>

   <td width="79"><em>σ </em>(sigma)</td>

  </tr>

  <tr>

   <td width="58">4</td>

   <td width="39">E</td>

   <td width="58">14</td>

   <td width="39">O</td>

   <td width="58">24</td>

   <td width="39">Y</td>

   <td width="58">4</td>

   <td width="79"><em>γ </em>(gamma)</td>

  </tr>

  <tr>

   <td width="58">5</td>

   <td width="39">F</td>

   <td width="58">15</td>

   <td width="39">P</td>

   <td width="58">25</td>

   <td width="39">Z</td>

   <td width="58">5</td>

   <td width="79"><em>δ </em>(delta)</td>

  </tr>

  <tr>

   <td width="58">6</td>

   <td width="39">G</td>

   <td width="58">16</td>

   <td width="39">Q</td>

   <td width="58"> </td>

   <td width="39"> </td>

   <td width="58">6</td>

   <td width="79"><em>λ </em>(lambda)</td>

  </tr>

  <tr>

   <td width="58">7</td>

   <td width="39">H</td>

   <td width="58">17</td>

   <td width="39">R</td>

   <td width="58"> </td>

   <td width="39"> </td>

   <td width="58">7</td>

   <td width="79"><em>ω </em>(omega)</td>

  </tr>

  <tr>

   <td width="58">8</td>

   <td width="39">I</td>

   <td width="58">18</td>

   <td width="39">S</td>

   <td width="58"> </td>

   <td width="39"> </td>

   <td width="58">8</td>

   <td width="79"><em>µ </em>(mu)</td>

  </tr>

  <tr>

   <td width="58">9</td>

   <td width="39">J</td>

   <td width="58">19</td>

   <td width="39">T</td>

   <td width="58"> </td>

   <td width="39"> </td>

   <td width="58">9</td>

   <td width="79"><em>ξ </em>(xi)</td>

  </tr>

 </tbody>

</table>

Each dataset is in .csv format, where each row is a data instance. Each instance is composed of 1024 binary features followed by its class (the index). Each dataset contains 3 splits:

<ul>

 <li><strong>training</strong>: to be used for training your models.</li>

 <li><strong>validation</strong>: to be used for validating/experimenting with your models.</li>

 <li><strong>test</strong>: to be used to report your final output.</li>

</ul>

<h1>2       Your Task</h1>

For each dataset, write the necessary code to:

<ol>

 <li>Plot the distribution of the number of the instances in each class.</li>

 <li>Run 6 different ML models:

  <ul>

   <li><strong>GNB: </strong>a Gaussian Naive Bayes Classifier, with default parameter values.</li>

   <li><strong>Base-DT: </strong>a baseline Decision Tree using entropy as decision criterion and using default values values for the rest of the parameters.</li>

   <li><strong>Best-DT: </strong>a better performing Decision Tree found by performing grid search to find the best combination of hyper-parameters. For this, you need to experiment with the following parameter values:

    <ul>

     <li>splitting criterion: gini and entropy</li>

     <li>maximum depth of the tree: 10 and no maximum</li>

     <li>minimum number of samples to split an internal node: experiment with values of your choice</li>

     <li>minimum impurity decrease: experiment with values of your choice</li>

     <li>class weight: None and balanced</li>

    </ul></li>

   <li><strong>PER: </strong>a Perceptron, with default parameter values..</li>

   <li><strong>Base-MLP: </strong>a baseline Multi-Layered Perceptron with 1 hidden layer of100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters.</li>

   <li><strong>Best-MLP: </strong>a better performing Multi-Layered Perceptron found by performing grid search to find the best combination of hyper-parameters. For this, you need to experiment with the following parameter values:

    <ul>

     <li>activation function: sigmoid, tanh, relu and identity</li>

     <li>2 network architectures of your choice: for eg 2 hidden layers with 30+50 nodes, 3 hidden layers with 10+10</li>

     <li>solver: Adam and stochastic gradient descent</li>

    </ul></li>

  </ul></li>

</ol>

<ol start="3">

 <li>For each model and each dataset, write the necessary code to generate a csv (comma separated values) output file that contains the output classification and the performance of each model for each dataset. This output file should be named [model name]-[dataset].csv. Therefore you should generate 12 files:</li>

</ol>

GNB-DS1 Base-DT-DS1 Best-DT-DS1 PER-DS1 Base-MLP-DS1 Best-MLP-DS1

GNB-DS1 Base-DT-DS2 Best-DT-DS1 PER-DS1 Base-MLP-DS2 Best-MLP-DS2

These files should contain:

<ul>

 <li>the row number of the instance, followed by a comma, followed by the index of the predicted class of that instance, as in:</li>

</ul>

1,24 // if your model’s predicted class for instance 1 is 24 (Y)

2,25 // if your model’s predicted class for instance 2 is 25 (Z)

3,4          // if your model’s predicted class for instance 3 is 4 (E)

<ul>

 <li>a plot the confusion matrix</li>

 <li>the precision, recall, and f1-measure for each class</li>

 <li>the accuracy, macro-average f1 and weighted-average f1 of the model</li>

</ul>

<h1>            3      Deliverables</h1>

The submission of the assignment will consist of 3 deliverables:

<ul>

 <li>The code &amp; output files</li>

 <li>The demo (8 min presentation &amp; Q/A)</li>

</ul>

<h2>            3.1       The Code &amp; Output files</h2>

Submit all files necessary to run your code in addition to a readme.md which will contain specific and complete instructions on how to run your experiments. You do not need to submit the datasets. If the instructions in your readme file do not work, are incomplete or a file is missing, you will not be given the benefit of the doubt.

Generate one output file for which model and each dataset test sets as indicated in Section 2.

<h2>            3.2     The Demos</h2>

You will have to demo your assignment for ≈ 12 minutes. Regardless of the demo time, you will demo the program that was uploaded as the official submission on or before the due date. The schedule of the demos will be posted on Moodle. The demos will consist in 2 parts: a presentation ≈ 8 minutes and a Q/A part (≈ 4 minutes). Note that the demos will be recorded.

<h3>             3.2.1       The Presentation</h3>

Prepare an 8-minute presentation to analyse and compare the performance of your models. The intended audience of your presentation is your TAs. Hence there is no need to explain the theory behind the models. Your presentation should focus on <strong>your </strong>work and the comparison of the performance of the models when the hyper-parameters are modified.

Your presentation should contain at least the following:

An analysis of the initial dataset given on Moodle. If there is anything particular about these datasets that might have an impact on the performance of some models, explain it.

An analysis of the results of all the models with the data sets. In particular, compare and contrast the performance of each model with one another, and with the datasets. Please note that your presentation must be analytical. This means that in addition to stating the facts (e.g. the macro-F1 has this value), you should also analyse them (i.e. explain why some metric seems more appropriate than another, or why your model did not do as well as expected. Tables, graphs and contingency tables to back up your claims would be very welcome here.

In the case of team work, a description of the responsibilities and contributions of each team member.

Any material used for the presentation (slides, …) must be uploaded on EAS before the due date.

<h3>             3.2.2      Q/A</h3>

After your presentation, your TA will proceed with a ≈ 4 minute question period. Each student will be asked questions on the code/assignment, and he/she will be required to answer the TA satisfactorily. In particular, each member should know what each parameters that you experimented with represent and their effect on the performance. Hence every member of team is expected to attend the demo.

In addition, your TA may give you a new dataset and ask you to train or run your models on this dataset. The output file generated by your program will have to be uploaded on EAS during your demo.

<h1>            4       Evaluation Scheme</h1>

Students in teams can be assigned different grades based on their individual contribution to project.

Individual grades will be based on:

<ul>

 <li>a peer-evaluation done after the submission.</li>

 <li>the contribution of each student as indicated on GitHub.</li>

 <li>the Q/A of each student during the demo.</li>

</ul>

The team grade will be based on:

<table width="629">

 <tbody>

  <tr>

   <td width="203">Code</td>

   <td width="400">functionality, proper use of the datasets, design, programming style, …</td>

   <td width="27">6</td>

  </tr>

  <tr>

   <td width="203">Output with initial datasets</td>

   <td width="400">correctness and format</td>

   <td width="27">1.5</td>

  </tr>

  <tr>

   <td width="203">Demo – Presentation</td>

   <td width="400">depth of the analysis, clarity and conciseness, presentation, time-management, …</td>

   <td width="27">4</td>

  </tr>

  <tr>

   <td width="203">Demo – QA</td>

   <td width="400">correct and clear answers to questions, knowledge of the program, …</td>

   <td width="27">2</td>

  </tr>

  <tr>

   <td width="203">Output with demo-dataset</td>

   <td width="400">correctness and format</td>

   <td width="27">1.5</td>

  </tr>

  <tr>

   <td width="203">Total</td>

   <td width="400"> </td>

   <td width="27">15</td>

  </tr>

 </tbody>

</table>





