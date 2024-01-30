## Description
News Classification Bot is an innovative tool that helps users manage news flow effectively. The bot uses advanced algorithms and machine learning techniques to automatically classify news into various categories.
The bot analyzes news headlines and content to determine their topic and relevance to the user. It can classify news into categories such as World, Sposts, Business and Ski/Tech

## Progress of the decision
Ag-news was chosen as the dataset. The dataset contains 120,000 records for the train sample and 7,600 for the validation sample. Labels: World, Sports, Business, Ski/Technology.
Two models were trained:
1.Full training bert-base-uncased;
2. Training roberta-base using LoRA.

## Results
The final metrics after fine-tuning of the two models differ slightly; the best accuracy and f1 values were shown by Bert-bass-uncased, but training with LoRA gives an advantage in time.
Due to the fact that the results are better for bert-bass-uncased, this model is used for answers in the bot.
