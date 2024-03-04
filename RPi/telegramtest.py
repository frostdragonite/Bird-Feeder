import time
from telegram.ext import Updater

updater = Updater("6172122051:AAHpsVfn2kNYL4apu4v2b99vscozCB8a9qw", use_context = True)

def send_to_telegram(filepath = "Test"):

        for attempt in range(10):
            try:
                print("Sending to telegram {0}".format(filepath))
                updater.bot.send_message(
                        chat_id="2084600136",
                        text="Hello Frost",
                )
                break
                # with open(filepath, "rb") as videofile:
                #     updater.bot.send_video(
                #         chat_id="2084600136",
                #         video=videofile,
                #         duration=60,
                #         caption="Motion detected!",
                #         supports_streaming=True,
                #         disable_notification=True,
                #         timout=30,
                #     )
                #     print("{0} sent successfully ".format(filepath))
                #     break
            except Exception as e:
                print(e)
                print(
                    'Module "%s": tg connection error {0}, retry.'.format(e)
                )
                time.sleep(5)

        return None
    
send_to_telegram()