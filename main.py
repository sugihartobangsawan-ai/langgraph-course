from dotenv import load_dotenv

load_dotenv()

from graph.graph import app
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
    print(
        app.invoke(
            input= {
                "question":"what is agent memory"
            }
        )
    )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
