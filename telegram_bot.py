import nn_module  # Модуль с нейронной сетью/ Module with the neural network
import telebot
from telebot import types

# Получение токена бота / Getting the bot token
#TOKEN = "TELEGRAM_BOT_TOKEN"

bot = telebot.TeleBot(TOKEN)

# Загрузка модели / Loading the model
#model_path = "models/model_eng.pt" #По-умолчанию / Default
#model = load_model(model_path)

settings = {'model_language': "eng"} #Модель для английского языка стоит по-умолчанию / Default model language is English

# Обработчик команды 'start' / Handler for the 'start' command
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.InlineKeyboardMarkup()
    lang_ru = types.InlineKeyboardButton('Русский', callback_data = 'russian')
    lang_eng = types.InlineKeyboardButton('Английский', callback_data = 'english')
    markup.row(lang_ru, lang_eng)
    bot.send_message(message.chat.id, 'Приветствую! На каком языке ваш текст?', parse_mode='html', reply_markup=markup) #Привет! Введите текст для предсказания


# Обработчик текстовых сообщений / Handler for text messages
@bot.message_handler()
def predict(message):
    prediction = nn_module.predict(message.text, settings['model_language'])
    bot.reply_to(message, prediction)

# Обработчик для callback/ Callback handler for language selection buttons
@bot.callback_query_handler(func = lambda callback: True)
def callback_mes(callback):
    if callback.data == 'russian':
        settings['model_language'] = "rus"
        bot.send_message(callback.message.chat.id, 'Выбрана модель для русского языка! Введите текст для предсказания уровня тяжести.', parse_mode='html')      #Russian is selected
    if callback.data == 'english':
        settings['model_language'] = "eng"
        bot.send_message(callback.message.chat.id, 'Выбрана модель для английского языка! Введите текст для предсказания уровня тяжести.', parse_mode='html')   #English is selected

# Запуск бота в режиме опроса / Running the bot in polling mode
bot.polling(none_stop=True)