import os
import telebot

from transformers import BertTokenizer, BertForSequenceClassification

bot = telebot.TeleBot('6500078265:AAEVj5DuR6XF6cdg1O_qaZHFkWll8wqlqPA')

@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, 'Greetings! This bot helps classify any news into 4 different classes: World, Sports, Business, Sci/Tech. You can check this by sending any news.')


@bot.message_handler(func=lambda message: True)
def handle_text(message):
    tokenizer = BertTokenizer.from_pretrained("./results/tokenizer")
    model = BertForSequenceClassification.from_pretrained("./results/model")

    news_categories=["World", "Sports", "Business", "Sci/Tech"]

    idx2cate = { i : item for i, item in enumerate(news_categories)}
    text = message.text

    max_length = 512
    inputs = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)

    label_index = probs.argmax(dim=1)[0].tolist()
    label = idx2cate[ label_index ]

    proba = round(float(probs.tolist()[0][label_index]),2)

    response = {'label': label, 'proba': proba}

    bot.reply_to(message, f'Response: {response}')

bot.infinity_polling()
