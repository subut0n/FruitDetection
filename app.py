from App.A_iFruits import app
import os 
from dotenv import load_dotenv
load_dotenv() 


if __name__ == "__main__":
    app.run(debug=False, port=os.getenv('PORT'))
