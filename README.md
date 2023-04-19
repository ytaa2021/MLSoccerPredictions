# Soccer Match Predictor - Introduction Rough Draft

<!-- ![](images/soccer-cartoon.gif) -->
<p align="center">
  <img width="460" height="300" src="images/soccer-cartoon.gif">
</p>

### Team members: Anna, Elshiekh, Alan, Yotam

## Introduction

If you follow soccer or any sport for that matter, religiously, you’ll quickly realize that the outcomes are as unpredictable as guessing lottery numbers. Furthermore, accurately predicting the outcome of soccer matches is difficult. Last year, 1.5 billion people watched the world cup, making soccer the most popular sport in the world. Fans around the world try to predict the outcome of soccer matches for fun or even to make money through sports betting. As such, it is important to create a machine-learning model that can accurately predict the outcome of soccer matches. 

You can always have a good idea of who might win based on team strength, but even that cannot be trusted at times due to the many other factors that come into play: inconsistency of players, unpredictable events during and surrounding a match, the dynamic nature of team performance and luck. Traditional statistical models lack the capability to incorporate various factors and rely heavily on past performance without considering current conditions. Furthermore, bias and limited capacity to analyze multiple factors in real-time add to the difficulty. Machine Learning has the potential to build relationships across many factors and potentially improve some of these difficulties.

Using an artificial neural network, we will use various factors such as player statistics, team dynamics, and performance in previous matches for our prediction. The integration of diverse data sources and the use of advanced machine learning algorithms will help overcome the limitations of prior work. The dataset used contains information about European soccer teams from 2008 to 2016. The Kaggle dataset contains information from 25,000+ matches and is a very population dataset for soccer. information.

In our results, we expect to see significant improvement in prediction accuracy compared to traditional statistical models and human intuition. We test our results by comparing our model’s predictions against actual match outcomes and against other prediction systems. In our future work, we would like to make regular updates to our model to keep it current with new information and features as well as potential advancements in machine learning techniques. 


## Related Works

There is a variety of studies that use neural networks to predict the outcomes of soccer matches. One such study is "Predicting Soccer Match Results in the English Premier League" by Ben Ulmer. The study aimed to predict the results of soccer matches in the English Premier League using machine learning models. The authors faced challenges such as the randomness of soccer data and the lack of injury information for players. Features such as whether a team is a home or away, team Elo rankings, and recent results were used for feature selection. The authors also tested various models including a stochastic gradient descent algorithm, Naive Bayes, a hidden Markov model, and a Support Vector Machine (SVM) with an RBF kernel. They managed to achieve about 50% accuracy when predicting whether the match was a win loss or a tie. The authors suggested that more advanced methods of feature selection and data preprocessing may improve the accuracy of the predictions. 

Another study that uses more common methods is "A deep learning framework for football [soccer] match predictions” by Ashiqur Rahman. This article, written by Ashiqur Rahman, focused on predicting the outcome of international soccer matches with a focus on the 2018 World Cup. The model was able to accurately predict 63.3% of outcomes correctly, however the author did mention that this number could be increased through more and accurate information on the teams. Some limitations to consider were that the model was able to accurately predict the majority of group stage matches, but failed quite often when it came to quarterfinal, semi-finals, and final matches; this may be because in the later stages teams tend to better matched up with each other, so the model was not be able to label a substantial difference in each team’s possibility to win.

A study that uses more advanced methods is “Predicting Wins, Losses and Attributes’ Sensitivities in the Soccer World Cup 2018 Using Neural Network Analysis” by Amr Hassan et. al. This paper analyzes data from the 2018 World Cup to create a model for predicting match results using supervised learning. The model is based on the “Radial basis function using 75 attributes”. The authors were able to achieve a win rate of 83.3% and a loss rate of 72.7% using key attributes such as Total Team Medium Pass Attempted (MBA) and the Distance Covered Team Average in zone 3. Additionally, the study “A Comparative Study on Neural Network Based Soccer Result Prediction” by Burak Galip Aslan and Mustafa Murat Inceoglu uses the black-box approach to achieve better prediction results. Traditional methods of predicting soccer matches rely on statistical models, however, the black box approach has proven to be better. The authors of the paper compared different models, including one by Cheng et. al. that uses a hybrid neural network, and their black box method was significantly better than the other methods. One thing the paper highlights is the importance of input data. More specifically, they note ‘The available data should be transformed into alternative formats… It may not be necessary to apply any available data in the form of input parameters’ (Aslan and Inceoglu). With this method, the authors of the paper were able to achieve correct home win predictions 70-80% of the time. Tiwari et. al, authors of "Football Match Result Prediction Using Neural Networks and Deep Learning" were able to achieve similar results. This paper uses data from the 2010-11 through 2017-18 seasons of the English Premier League. It employs recurrent neural networks with LSTM cell modifications to predict the outcome of soccer matches. This allowed them to achieve success percentage of 80.75, which is a 10% increase from using ANN.

“Neural Networks Football Result Prediction” by E. Tjioe , F. Syakir , R. H. C. Shum, I. Buo, students at the University of Tartu, used their model to test betting returns. They used data from the English Premier League and the Spanish La Liga. They first created baseline models using other machine learning algorithms, like logistic regression and random forests. They then built a neural network and tuned its hyperparameters to predict the winner of each game. They showed that betting for winner, home or away, and predicted winner with threshold returned greater profit than betting with draw only factors.

## Methods
We obtained the dataset from kaggle and need to make sure it looks clean and take care of some preprocessing logistics. Of the 25,000 data points collected from different leagues, we used matches from countries in the top 5 leagues: England, France, Germany, Italy, and Spain. 

We also removed data that was formatted in XML and the predicted odds from various betting companies. Our dataset then contains the ids for the country, home and away team, total goals scored, home wins and away wins percentage, and a column of 0s and 1s where 1s represent a home team win. We created 2 different dataframes, a simple 1 which only contains the home and away teams ID numbers, and the amount of goals scored by each team. The complex dataframe also contains the ID numbers, with the addion of each team's head-to-head history. The head-to-head history contains the total amount of times each team has played eachother, as well as the number of times the home and away team has won. Also included is the number of times the match has ended in a draw.

We are using pyTorch’s Neural Network library, from which we will most likely be using the feed forward architecture. We will be creating different models and comparing the accuracy of using different features. The models we created are labeled simple and complex. The simple model is a feedforward model with 3 layers. The input layer has 2 input nodes and 16 output nodes. the hidden layer contains 16 input and 32 output nodes. The output layer contains 32 input and 1 output. The complex model is also a feedforward model with 3 layers. The layers are identical, except the input layer has 4 input nodes. 

After the dataframes were completed, we normalized the data using a label encoder, and then split the data into a training and test set. For the simple model, the training set was made up of the home and away team's ID numbers, and the test set is a vector of 0s and 1s, where a 1 represents a home team win. For the complex model, the training set included the draw count and number of total matches played, in additional to both team's ID number. The test set is idential, except a 1 represents that the home team has more wins than the away team.

For our model, we are hoping to train the neural network to accurately predict whether Team A or Team B will win, or if the result will be a tie.

The end result we are hoping for is a model that when asked for the outcome of a soccer match would seem like a very well informed sports analyst.

Possible pitfalls we see in our model is low or inaccurate classification.


## Discussion
We used the European Soccer Database and filtered out the teams that did not belong to England, France, Italy, Spain, or Germany. We created 2 models, a simple feedforward neural network that only considers the goals scored by both teams and a more complex model that factors in home win percentage, face-to-face history, and player stats. It should be noted that player stats are often not a good measure for predicting the outcome. Roland Shum in Neural Networks Football Result Prediction points out that player morale, a stat that cannot be measured, often plays a role in determining how a player does in a given match. Our comparison will focus on the accuracy of these models to those of other models built. Running the simple model produces an accuracy of ~55%. Compared to our related works our results were significantly worse. We yielded an accuracy of below 60% for our validation dataset while many optimized models in the field can consistently predict with above 70% accuracy. Looking at fig. 1, 
<!-- ![Figure 1. Simple Loss Graph](SimpleGraph.png) -->
<p align="center">
  <img width="460" height="300" src="images/SimpleGraph.png">
</p>
one can see that validation loss is lower than training loss, which indicates that our model performs slighly better on data it has not seen before. The complex model yielded slightly better results, with an accuracy of 59%. Fig. 2 shows the loss over epoch graph. 
<!-- ![Figure 2. Complex Loss Graph](ComplexGraph.png) -->
<p align="center">
  <img width="460" height="300" src="images/ComplexGraph.png">
</p>
This time, our model worked better on data it has seen before.  We are intending on using a classification model architecture and building different combinations of features for further experiments. A function was also created that takes as input the ID number for two teams, and using the complex  model, predicts the winner. However the function always predicted the same accuracy even when the teams were switched. 

## Ethics
Predicting sporting outcomes does not cause any harm. However, harm may occur when individuals use this model to gamble on sports betting websites. The European Soccer Database contains columns that list the betting odds  from a variety of sports betting websites. While listing the betting odds may be useful to gain insight on the prediction made by the company, we want to avoid incentivizing the urge to gamble.The one thing we want to avoid is a user using our model to gamble on soccer matches, and potentially losing money. Cumpulsive Gambling is a serious problem that has real word affects on an individual and their family. 

## Reflection
We would’ve liked to use a dataset that is more recent than what we used as our dataset only spans 2008 to 2016. Soccer teams can significantly evolve year-to-year, so our model likely wouldn’t be very useful for predicting current or future matches. We also would have used a dataset with more information – perhaps one that also contains individual player information. Using this, we could track which teams players are on and use their information to help predict how good a team might be. 

If presented with more time, we would’ve like to add a feature where instead of just picking the winning team, it could also predict the score line. Furthermore, it could predict who was likely to score based on individual players’ previous performances and their current up to date form. 

We would of also like to have a dynamically updating dataset that instead of using data from a fixed number of years (e.g. 2012-2018), it would instead take the data from the previous 5 years of the current year (e.g. if its 2023, the data used will be from 2017-2022). 

Additionally we believe that more features would allow us to achieve a higher accuracy. We are curious to do an analysis on the relevance of different features for the project to inform us as a heuristic for predicting matches, and just to obtain larger insights about the sport in general.

Our data focused more so on league games, which are a bit more predictable due to teams having a lot of head to head matchups against each other. We want to continue this work by expanding the scope of possible games through including more European based games (teams from different countries such as Real Madrid vs. Chelsea)  such as from the Champions League, Europa League, etc. 


## Works Cited
B. G. Aslan and M. M. Inceoglu, "A Comparative Study on Neural Network Based Soccer Result Prediction," Seventh International Conference on Intelligent Systems Design and Applications (ISDA 2007), Rio de Janeiro, Brazil, 2007.

E. Tiwari, P. Sardar and S. Jain, "Football Match Result Prediction Using Neural Networks and Deep Learning," 2020 8th International Conference on Reliability, Infocom Technologies and Optimization (Trends and Future Directions) (ICRITO). 2020.

Hassan A, Akl A-R, Hassan I, Sunderland C. Predicting Wins, Losses and Attributes’ Sensitivities in the Soccer World Cup 2018 Using Neural Network Analysis. Sensors. 2020.

Rahman, M.A. A deep learning framework for football match prediction. (2020)

Shum, Roland. "Neural Networks Football Result Prediction" 2020.

Ulmer, Benjamin and M. Pasadas Fernández. “Predicting Soccer Match Results in the English Premier League.” 2014.
