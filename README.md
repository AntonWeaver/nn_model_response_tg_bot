# Response app for NN model - via Telegram bot

This is response App for developed NN model. The main idea behind is the same as github.com/AntonWeaver/nn_model_response, but final implementation is different.
The main idea - create a simple app to get some response from NN model: input text to Telegram bot->  request processing  through model -> text output from the model to telegram bot for User.

Libs used: telebot, torch, transformers 

The project includes:

- Working with telebot library
- Back-end to get responce from the model
- Result check via telegram

A simple model selection is also implemented, depending on the user’s needs.

TODO:

- Implement async in tg-bot
- Try telegram lib
