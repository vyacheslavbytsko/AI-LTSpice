from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import os
import tempfile
from voice import process_voice_message, text_to_speech

# Токен вашего бота
BOT_TOKEN = "7577700732:AAH4UjUxOCUgFDmsFcrQr4anOqcUJp4QzYc"

async def start(update: Update, context):
    await update.message.reply_text("Привет! Отправь мне голосовое сообщение, и я преобразую его в текст.")

async def handle_voice_message(update: Update, context):
    # Скачиваем голосовое сообщение
    voice_file = await update.message.voice.get_file()
    with tempfile.NamedTemporaryFile(suffix=".oga", delete=False) as temp_oga_file:
        oga_path = temp_oga_file.name
        await voice_file.download_to_drive(oga_path)

    # Обрабатываем голосовое сообщение
    text = process_voice_message(oga_path)

    # Удаляем временный .oga файл
    os.remove(oga_path)

    # Отправляем распознанный текст пользователю
    await update.message.reply_text(f"Распознанный текст: {text}")

    # Преобразуем текст обратно в аудио и отправляем
    audio_path = text_to_speech(text)
    await update.message.reply_voice(voice=open(audio_path, 'rb'))

    # Удаляем временный .mp3 файл
    os.remove(audio_path)

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Регистрируем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    # Запускаем бота
    app.run_polling()

if __name__ == "__main__":
    main()