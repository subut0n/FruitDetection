from App.A_iFruits import app
import os 
from dotenv import load_dotenv
load_dotenv() 


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=os.getenv('PORT'))
