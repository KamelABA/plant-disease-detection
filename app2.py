from flask import Flask, render_template, request, redirect, send_from_directory, session, url_for
import cv2
import tensorflow as tf
import numpy as np
import datetime
from keras.models import load_model
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def calculate_mean(arr: np.ndarray) -> float:
    return np.mean(arr)


app = Flask(__name__, template_folder='HtmlPage')
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')

# MongoDB Atlas Connection
# IMPORTANT: Set MONGODB_URI in your .env file - DO NOT commit credentials to git
MONGODB_URI = os.getenv('MONGODB_URI')

if not MONGODB_URI:
    raise ValueError("❌ MONGODB_URI environment variable is not set. Please create a .env file with your MongoDB connection string.")

try:
    client = MongoClient(MONGODB_URI)
    db = client['PlantDisease']
    
    # Collections (equivalent to SQLite tables)
    users_collection = db['users']
    disease_info_collection = db['DiseaseInformation']
    supplement_collection = db['Supplement']
    historique_collection = db['Historique']
    
    # Test connection
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"❌ Error connecting to MongoDB Atlas: {e}")
    raise e

COUNT = 0

model_potato = load_model("Potato.h5")
model_tomato = load_model("Tomato.h5")
model_grape = load_model("Grape.h5")
model_cotton = load_model("Cotton.h5")
model_corn = load_model("Corn.h5")
# model_Apple = load_model("apple.h5")



HtmlTemplate = ['potato', 'cotton', 'corn', 'tomato', 'grape','Apple']


@app.route('/')
def home():
    return render_template('home.html')




@app.route('/login')
def Login():
     error = request.args.get('error')
     return render_template('login.html', error=error)

@app.route('/detect_plant')
def detect_plant():

    if 'email' in session and 'id' in session:
        email = session['email']
        password = session['password']
        user_id = session['id']
        
    try:
        # Check if the email and password exist in the users collection
        user = users_collection.find_one({'email': email, 'password': password})
    except Exception as e:
        return redirect(url_for('Login', error=str(e)))
    
    if user:
        session['email'] = email
        session['password'] = password
        session['id'] = str(user['_id'])
        return redirect(url_for('leaf'))
    else:
        return redirect(url_for('Login', error='User not found or incorrect password.', email=email, password=password, id=user_id))



visitor_count = 0

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    global visitor_count
    visitor_count += 1  # Increment the visitor count on each visit

    # Handle POST request for language selection
    if request.method == 'POST':
        lang = request.form.get('lang')  # Get selected language from the form
        session['lang'] = lang  # Store the selected language in the session
        return redirect(url_for('settings'))  # Redirect to reflect the changes

    # Default language is English if not set in the session
    lang = session.get('lang', 'en')

    return render_template('settings.html', lang=lang, visitor_count=visitor_count)

@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/users')
def users():
    try:
        # Fetch all user data from MongoDB
        users_data = list(users_collection.find({}, {'_id': 1, 'email': 1, 'password': 1}))
        
        # Convert to list of tuples for template compatibility
        users_list = [(str(user['_id']), user.get('email', ''), user.get('password', '')) for user in users_data]
        
        return render_template('users.html', users=users_list)
    except Exception as e:
        return str(e)


@app.route('/supplement')
def supplement():
    try:
        # Fetch all supplement data from MongoDB
        req_data = list(supplement_collection.find({}, {'_id': 1, 'disease_name': 1, 'supplementName': 1}))
        
        # Convert to list of tuples for template compatibility
        req_list = [(str(item['_id']), item.get('disease_name', ''), item.get('supplementName', '')) for item in req_data]
        
        return render_template('supplement.html', req=req_list)
    except Exception as e:
        return str(e)


@app.route('/diseaseinfo')
def diseaseinfo():
    try:
        # Fetch all disease info from MongoDB
        req_data = list(disease_info_collection.find({}, {'_id': 1, 'disease_name': 1, 'description': 1, 'PossibleSteps': 1}))
        
        # Convert to list of tuples for template compatibility
        cleaned_data = []
        for item in req_data:
            _id = str(item['_id'])
            disease_name = item.get('disease_name', '')
            description = item.get('description', '')
            possible_steps = item.get('PossibleSteps', '')
            
            # Handle encoding issues
            if isinstance(possible_steps, bytes):
                possible_steps = possible_steps.decode('utf-8', 'replace')
            else:
                possible_steps = str(possible_steps)
                
            cleaned_data.append((_id, disease_name, description, possible_steps))

        return render_template('diseasesInfo.html', req=cleaned_data)
    except Exception as e:
        return str(e)


@app.route('/<temp>')
def predict_crop(temp):
    if temp in HtmlTemplate:
        return render_template(f'{temp}.html')
    else:
        return render_template('404.html'), 404


def get_disease_info(disease_name):
    result = disease_info_collection.find_one({'disease_name': disease_name})
    
    if result:
        Description = result.get('description', 'No information available')
        possiblesteps = result.get('PossibleSteps', 'No recommendations')
    else:
        Description, possiblesteps = "No information available", "No recommendations"
    return Description, possiblesteps


def get_Supplement_info(disease_name):
    result = supplement_collection.find_one({'disease_name': disease_name})
    
    if result:
        SupplementName = result.get('SupplementName', result.get('supplementName', 'No information available'))
        SupplementImage = result.get('SupplementImage', 'No recommendations')
    else:
        SupplementName, SupplementImage = "No information available", "No recommendations"
    return SupplementName, SupplementImage


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/adduser', methods=['POST'])
def adduser():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        name = request.form['name']
        try:
            # Insert user into MongoDB
            result = users_collection.insert_one({
                'email': email,
                'password': password,
                'name': name,
                'created_at': datetime.datetime.now()
            })
            
            return redirect(url_for('Login'))
        except Exception as e:
            error_message = f"Error occurred while inserting data: {e}. Details: {str(e)}"
            return error_message

    return redirect(url_for('signup'))


@app.route('/leaf-detection')
def leaf():
    if 'email' in session and 'id' in session:
        email = session['email']
        user_id = session['id']
        
        # Fetch the name from MongoDB
        try:
            user = users_collection.find_one({'_id': ObjectId(user_id)})
            name = user.get('name', None) if user else None
        except:
            # If user_id is not a valid ObjectId, try email lookup
            user = users_collection.find_one({'email': email})
            name = user.get('name', None) if user else None

        return render_template('leaf-detection.html', name=name, email=email, id=user_id)
    else:
        return "Session data missing."





@app.route("/checklogin", methods=['POST'])
def checklogin():
    email = request.form['email']
    password = request.form['password']
    
    try:
        # Check if user exists in MongoDB
        user = users_collection.find_one({'email': email, 'password': password})
    except Exception as e:
        return redirect(url_for('Login', error=str(e)))
    
    if user:
        session['email'] = email
        session['password'] = password
        session['id'] = str(user['_id'])
        return redirect(url_for('leaf'))
    else:
        return redirect(url_for('Login', error='User not found or incorrect password.'))


def save_image(img, count):
    img_path = f'static/img/{count}.jpg'
    img.save(img_path)
    if not os.path.exists(img_path):
        raise ValueError("Image not saved correctly")
    return img_path

def preprocess_image(img_path):
    img_arr = cv2.imread(img_path)
    if img_arr is None:
        raise ValueError("Image not read correctly")
    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    return img_arr

def store_prediction(user_id, disease_name, description, possiblesteps, supplement_name, current_date):
    historique_collection.insert_one({
        'iduser': user_id,
        'DiseaseName': disease_name,
        'Explanation': description,
        'Recommendations': possiblesteps,
        'Supplement': supplement_name,
        'date': current_date
    })

def predict_disease(model, get_disease_func, endpoint):
    global COUNT
    img = request.files['image']
    if img.mimetype not in ['image/jpeg', 'image/png']:
        return render_template('Error.html', message="Unsupported image format. Please upload a JPEG or PNG image.")

    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    try:
        img_path = save_image(img, COUNT)
        img_arr = preprocess_image(img_path)
        predictions = model.predict(img_arr)
        
        # Use .any() to check array is not empty
        if predictions.size == 0:
            raise ValueError("No prediction could be made")

        # Use np.argmax correctly to get prediction
        prediction = np.argmax(predictions, axis=1)[0]
        COUNT += 1

        disease_info = get_disease_func(prediction)
        disease_name, color, Description, possiblesteps, SupplementName, SupplementImage = disease_info

        store_prediction(session['id'], disease_name, Description, possiblesteps, SupplementName, current_date)
    except Exception as e:
        return render_template('Error.html', message=str(e))

    return render_template('Output.html', data=(disease_name, color), img_path=img_path, Description=Description, possiblesteps=possiblesteps, SupplementName=SupplementName, SupplementImage=SupplementImage)

@app.route('/predictionpotato', methods=['POST'])
def prediction_potato():
    return predict_disease(model_potato, get_potato_disease, 'potato')

@app.route('/predictiontomato', methods=['POST'])
def prediction_tomato():
    return predict_disease(model_tomato, get_tomato_disease, 'tomato')

@app.route('/predictiongrape', methods=['POST'])
def prediction_grape():
    return predict_disease(model_grape, get_grape_disease, 'Grape')

@app.route('/predictioncotton', methods=['POST'])
def prediction_cotton():
    return predict_disease(model_cotton, get_cotton_disease, 'cotton')

@app.route('/predictioncorn', methods=['POST'])
def prediction_corn():
    return predict_disease(model_corn, get_corn_disease, 'corn')

@app.route('/predictionapple', methods=['POST'])
def prediction_apple():
    return predict_disease(model_Apple, get_apple_disease, 'Apple')

def get_potato_disease(prediction):
    if prediction == 0:
        return [
            "Potato Early Blight", 'red',
            *get_disease_info("Potato : Early Blight"),
            *get_Supplement_info("Potato___Early_blight")
        ]
    elif prediction == 1:
        return [
            "Potato Late Blight", 'red',
            *get_disease_info("Potato : Late Blight"),
            *get_Supplement_info("Potato___Late_blight")
        ]
    elif prediction == 2:
        return [
            "Potato Healthy", 'green',
            *get_disease_info("Potato : Healthy"),
            *get_Supplement_info("Potato___healthy")
        ]

def get_tomato_disease(prediction):
    if prediction == 0:
        return [
            "Bacterial Spot", 'red',
            *get_disease_info("Tomato : Bacterial Spot"),
            *get_Supplement_info("Tomato : Bacterial Spot")
        ]
    elif prediction == 1:
        return [
            "Early Blight", 'red',
            *get_disease_info("Tomato : Early Blight"),
            *get_Supplement_info("Tomato : Early Blight")
        ]
    elif prediction == 2:
        return [
            "Late Blight", 'red',
            *get_disease_info("Tomato : Late Blight"),
            *get_Supplement_info("Tomato : Late Blight")
        ]
    elif prediction == 3:
        return [
            "Leaf Mold", 'red',
            *get_disease_info("Tomato : Leaf Mold"),
            *get_Supplement_info("Tomato : Leaf Mold")

        ]
    elif prediction == 4:
        return [
        "Septoria Leaf Spot", 'red',
            *get_disease_info("Tomato : Septoria Leaf Spot"),
            *get_Supplement_info("Tomato : Septoria Leaf Spot")

        ]
    elif prediction == 5:
        return [
        "Spider Mites", 'red',
            *get_disease_info("Tomato : Spider Mites | Two-Spotted Spider Mite"),
            *get_Supplement_info("Tomato : Spider Mites | Two-Spotted Spider Mite")

        ]
    elif prediction == 6:
        return [
        "Target Spot", 'red',
            *get_disease_info("Tomato : Target Spot"),
            *get_Supplement_info("Tomato : Target Spot")

        ]
    elif prediction == 7:
         return [
        "Yellow Leaf Curl Virus", 'red',
            *get_disease_info("Tomato : Yellow Leaf Curl Virus"),
            *get_Supplement_info("Tomato : Yellow Leaf Curl Virus")

        ]
    elif prediction == 8:
         return [
        "Tomato Mosaic Virus", 'red',
            *get_disease_info("Tomato : Mosaic Virus"),
            *get_Supplement_info("Tomato : Mosaic Virus")

        ]
    else:
        return [
            "Tomato Healthy", 'green',
            *get_disease_info("Tomato : Healthy"),
            *get_Supplement_info("Tomato : Healthy")
        ]

def get_grape_disease(prediction):
    if prediction == 0:
        return[
            "Grape Black rot","red",
            *get_disease_info("Grape : Black Rot"),
            *get_Supplement_info("Grape : Black Rot")

        ]
    elif prediction ==1:
        return[
            "Grape Esca(Black_Measles)","red",
            *get_disease_info("Grape : Esca | Black Measles"),
            *get_Supplement_info("Grape : Esca | Black Measles")


        ]
    elif prediction ==2:
        return[
        "Grape Leaf blight (Isariopsis Leaf Spot)","red",
            *get_disease_info("Grape : Leaf Blight | Isariopsis Leaf Spot"),
            *get_Supplement_info("Grape : Leaf Blight | Isariopsis Leaf Spot")

        ]
    else:
        return[
            "Grape Healthy", 'green',
            *get_disease_info("Grape : Healthy"),
            *get_Supplement_info("Grape : Healthy")
        ]

def get_corn_disease(prediction):
    if prediction == 0:
        return[
            "Blight", 'red',
            *get_disease_info("Corn : Northern Leaf Blight"),
            *get_Supplement_info("Corn : Northern Leaf Blight")
        ]
    elif prediction == 1:
        return[
            "Common Rust", 'red',
            *get_disease_info("Corn : Common Rust"),
            *get_Supplement_info("Corn : Common Rust")

        ]
    elif prediction ==2:
        return[
            "Corn : Cercospora Leaf Spot | Gray Leaf Spot","red",
            *get_disease_info("Corn___Cercospora_leaf_spot Gray_leaf_spot"),
            *get_Supplement_info("Corn___Cercospora_leaf_spot Gray_leaf_spot")

        ]
    else:
        return [
            "Corn Healthy", 'green',
            *get_disease_info("Corn : Healthy"),
            *get_Supplement_info("Corn : Healthy")
        ]

def get_cotton_disease(prediction):
    if prediction ==0:
        return[
            "diseased cotton leaf","red",
            "Cotton leaf disease detected. The leaf shows signs of infection.",
            "Remove affected leaves, apply appropriate fungicide, ensure proper drainage.",
            "Copper-based fungicide",
            ""
        ]
    elif prediction ==1:
        return[
            "diseased cotton plant", "red",
            "Cotton plant disease detected. The plant shows signs of infection.",
            "Isolate affected plants, apply systemic fungicide, improve air circulation.",
            "Systemic fungicide",
            ""
        ]
    elif prediction ==2:
        return[
            "fresh cotton leaf","green",
            "The cotton leaf is healthy with no signs of disease.",
            "Continue regular care and monitoring.",
            "No supplement needed",
            ""
        ]
    else : 
         return[
            "fresh cotton plant","green",
            "The cotton plant is healthy with no signs of disease.",
            "Continue regular care and monitoring.",
            "No supplement needed",
            ""
        ]
    
def get_apple_disease(prediction):
    if prediction == 0:
        return[
            "Apple Scab","red",
            *get_disease_info("Apple : Scab"),
            *get_Supplement_info("Apple : Scab")

        ]
    elif prediction ==1:
        return[
            "Apple Black rot","red",
            *get_disease_info("Apple : Black rot"),
            *get_Supplement_info("Apple : Black rot")


        ]
    elif prediction == 2:
        return[
        "Apple Cedar rust","red",
            *get_disease_info("Apple : Cedar rust"),
            *get_Supplement_info("Apple : Cedar rust")

        ]
    else:
        return[
            "Apple Healthy", 'green',
            *get_disease_info("Apple : Healthy"),
            *get_Supplement_info("Apple : Healthy")
        ]
    
def get_visitor_ip():
    if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
        return request.environ['REMOTE_ADDR']
    else:
        return request.environ['HTTP_X_FORWARDED_FOR']

@app.route('/index')
def index():
    visitor_ip = get_visitor_ip()
    user_agent = request.headers.get('User-Agent')
    
    # Log visitor info (in this example, print to console or save to a file)
    log_visitor_info(visitor_ip, user_agent)
    
    return f"Welcome! Your IP: {visitor_ip}, Your User-Agent: {user_agent}"

# Log visitor info to console (can be modified to log to a file or database)
def log_visitor_info(ip, user_agent):
    print(f"Visitor IP: {ip}, User-Agent: {user_agent}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
