from App.A_iFruits import app
import os 
from dotenv import load_dotenv
load_dotenv() 


if __name__ == "__main__":
    app.run(debug=True, host="https://foodexv1.azurewebsites.net", port=os.getenv('PORT'))
