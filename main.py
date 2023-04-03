from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from keras.models import load_model


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
array = ["Bacterial Spot",
    "Early_Blight",
    "Late_Blight",
    "Leaf Mold",
    "Septoria Leaf",
    "Spider Mites",
    "Target spot",
    "Tomato yellow leaf curl virus",
    "Tomato Mosaic virus",
    "Tomato Healthy"]

model3 = load_model("Models/PIL_trained_.h5", compile=False)
model3.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model2 = load_model("Models/potato.h5", compile=False)
model2.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model4 = load_model("Models/cauli.h5", compile=False)
model4.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model5 = load_model("Models/okra.h5", compile=False)
model5.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
model6 = load_model("Models/cabbage.h5", compile=False)
model6.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model1 = load_model("Models/casava2.h5", compile=False)
model1.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])


def read_img(data):
    image = Image.open(BytesIO(data))
    image = image.resize((80, 80))
    image = np.array(image)
    return image


@app.get("/", response_class=HTMLResponse)
async def ping():
    return """
 <!DOCTYPE html>
<html>
<head>
	<title>Vegetable Buttons</title>
	<style>
		body {
			display: flex;
			flex-direction: column;
			align-items: center;
			justify-content: center;
			height: 100vh;
			margin: 0;
			padding: 0;
			background: url(https://w0.peakpx.com/wallpaper/837/398/HD-wallpaper-yellow-green-field-during-sunset-field-sunset-grass-nature-graphy.jpg) no-repeat;
            background-position: center;
            background-size: cover;
            min-height: 100vh;
            width: 100%;
		}
		.btn-group {
			display: flex;
			flex-wrap: wrap;
			justify-content: center;
			margin-bottom: 30px;
		}
		.btn {
			display: inline-block;
			margin: 10px;
			padding: 30px 40px;
			background-color: transparent;
			color: rgb(15, 14, 14);
			border: none;
			border-radius: 10px;
			cursor: pointer;
			transition: all 0.2s ease-in-out;
		}
		.btn:hover {
			transform: scale(1.2);
			box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
		}

		a:link {
  color: rgb(0, 0, 0);
  background-color: transparent;
  text-decoration: none;
}

a:visited {
	color: rgb(0, 0, 0);
  background-color: transparent;
  text-decoration: none;
}

a:hover {
	color: rgb(0, 0, 0);
  background-color: transparent;
  text-decoration: underline;
}

a:active {
	color: rgb(0, 0, 0);
  background-color: transparent;
  text-decoration: underline;
}
	</style>
</head>
<body>
	<h1>Choose Vegetable</h1>
	<div class="btn-group">
		<button class="btn"><a href="/upload_cassava"><h1>Cassava</h1></a></button>
		<button class="btn"><a href="/upload_potato"><h1>Potato</h1></a></button>
		<button class="btn"><a href="/upload_tomato"><h1>Tomato</h1></a></button>
		<button class="btn"><a href="/upload_cauliflower"><h1>Cauliflower</h1></a></button>
		<button class="btn"><a href="/upload_okra"><h1>Okra</h1></a></button>
		<button class="btn"><a href="/upload_cabbage"><h1>Cabbage</h1></a></button>
	</div>
	<h3>This is an AI based website, which can predict various diseases related to commonly grown crops.<br>
	 User can upload image of the crops which user have selected and upload. Our AI models will predict the <br>
	 likelihood of a disease with it's confidence.</h3>
	<script>
		// code for button hover effect
		const buttons = document.querySelectorAll('.btn');
		buttons.forEach(btn => {
			btn.addEventListener('mouseover', () => {
				btn.style.transform = 'scale(1.2)';
				btn.style.boxShadow = '0 0 10px rgba(0, 0, 0, 0.5)';
			});
			btn.addEventListener('mouseout', () => {
				btn.style.transform = 'scale(1)';
				btn.style.boxShadow = 'none';
			});
		});
	</script>
</body>
</html>
    """


@app.get("/upload_cassava")
async def upload_page():
    content="""<!DOCTYPE html>
        <html>
        <head>
            <title>Image Upload Page</title>
            <style>
    .center-box {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.upload-box {
  border: 1px solid #ccc;
  padding: 10px;
  margin-bottom: 10px;
}

.upload-btn {
  display: block;
  width: 100px;
  height: 100px;
  font-size: 16px;
  padding: 10px;
  background-color: #4CAF50;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin: auto;
}
    </style>
        </head>
        <body>
             <div class="center-box">
        <form action="/predict_cassava" method="POST" enctype="multipart/form-data">
          <div class="upload-box">
            <label for="image-upload">Upload an image:</label>
            <input type="file" id="image-upload" name="image">
          </div>
          <button type="submit"> upload image</button>
        </form>
      </div>
        </body>
        </html>"""
    return HTMLResponse(content=content)

@app.post("/predict_cassava")
async def predict(image: UploadFile = File(...)):
        array = ["Cassava Bacterial Blight (CBB)",
                           "Cassava Brown Streak Disease (CBSD)",
                           "Cassava Green Mottle (CGM)",
                           "Cassava Mosaic Disease (CMD)",
                           "Healthy"]
        img = read_img(await image.read())
        img = img / 255
        # print(img)
        img = np.expand_dims(img, axis=0)
        prediction = model1.predict(img)
        return {
            "Prediction": array[np.argmax(prediction[0]).tolist()],
            "Surity": str(np.max(prediction).tolist()*100) + " %"
        }


@app.get("/upload_potato")
async def upload_page():
    content="""<!DOCTYPE html>
        <html>
        <head>
            <title>Image Upload Page</title>
            <style>
    .center-box {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.upload-box {
  border: 1px solid #ccc;
  padding: 10px;
  margin-bottom: 10px;
}

.upload-btn {
  display: block;
  width: 100px;
  height: 100px;
  font-size: 16px;
  padding: 10px;
  background-color: #4CAF50;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin: auto;
}
    </style>
        </head>
        <body>
             <div class="center-box">
        <form action="/predict_potato" method="POST" enctype="multipart/form-data">
          <div class="upload-box">
            <label for="image-upload">Upload an image:</label>
            <input type="file" id="image-upload" name="image">
          </div>
          <button type="submit"> upload image</button>
        </form>
      </div>
        </body>
        </html>"""
    return HTMLResponse(content=content)

@app.post("/predict_potato")
async def predict(image: UploadFile = File(...)):
        array = ["Early Blight", "Healthy", 'Late Blight']
        img = read_img(await image.read())
        img = img / 255
        # print(img)
        img = np.expand_dims(img, axis=0)
        prediction = model2.predict(img)
        return {
            "Prediction": array[np.argmax(prediction[0]).tolist()],
            "Surity": str(np.max(prediction).tolist()*100) + " %"
        }


@app.get("/upload_tomato")
async def upload_page():
    content="""<!DOCTYPE html>
        <html>
        <head>
            <title>Image Upload Page</title>
            <style>
    .center-box {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.upload-box {
  border: 1px solid #ccc;
  padding: 10px;
  margin-bottom: 10px;
}

.upload-btn {
  display: block;
  width: 100px;
  height: 100px;
  font-size: 16px;
  padding: 10px;
  background-color: #4CAF50;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin: auto;
}
    </style>
        </head>
        <body>
             <div class="center-box">
        <form action="/predict_tomato" method="POST" enctype="multipart/form-data">
          <div class="upload-box">
            <label for="image-upload">Upload an image:</label>
            <input type="file" id="image-upload" name="image">
          </div>
          <button type="submit"> upload image</button>
        </form>
      </div>
        </body>
        </html>"""
    return HTMLResponse(content=content)

@app.post("/predict_tomato")
async def predict(image: UploadFile = File(...)):
        array = ["Bacterial Spot",
        "Early_Blight",
        "Late_Blight",
        "Leaf Mold",
        "Septoria Leaf",
        "Spider Mites",
        "Target spot",
        "Tomato yellow leaf curl virus",
        "Tomato Mosaic virus",
        "Tomato Healthy"]
        img = read_img(await image.read())
        img = img / 255
        # print(img)
        img = np.expand_dims(img, axis=0)
        prediction = model3.predict(img)
        return {
            "Prediction": array[np.argmax(prediction[0]).tolist()],
            "Surity": str(np.max(prediction).tolist()*100) + " %"
        }


@app.get("/upload_cauliflower")
async def upload_page():
    content="""<!DOCTYPE html>
        <html>
        <head>
            <title>Image Upload Page</title>
            <style>
    .center-box {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.upload-box {
  border: 1px solid #ccc;
  padding: 10px;
  margin-bottom: 10px;
}

.upload-btn {
  display: block;
  width: 100px;
  height: 100px;
  font-size: 16px;
  padding: 10px;
  background-color: #4CAF50;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin: auto;
}
    </style>
        </head>
        <body>
             <div class="center-box">
        <form action="/predict_cauliflower" method="POST" enctype="multipart/form-data">
          <div class="upload-box">
            <label for="image-upload">Upload an image:</label>
            <input type="file" id="image-upload" name="image">
          </div>
          <button type="submit"> upload image</button>
        </form>
      </div>
        </body>
        </html>"""
    return HTMLResponse(content=content)

@app.post("/predict_cauliflower")
async def predict(image: UploadFile = File(...)):
        array = ['Bacterial spot rot' ,"Black Rot" ,"Downy Mildew" ,
                       "Healthy"]
        img = read_img(await image.read())
        img = img / 255
        # print(img)
        img = np.expand_dims(img, axis=0)
        prediction = model4.predict(img)
        return {
            "Prediction": array[np.argmax(prediction[0]).tolist()],
            "Surity": str(np.max(prediction).tolist()*100) + " %"
        }


@app.get("/upload_okra")
async def upload_page():
    content="""<!DOCTYPE html>
        <html>
        <head>
            <title>Image Upload Page</title>
            <style>
    .center-box {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.upload-box {
  border: 1px solid #ccc;
  padding: 10px;
  margin-bottom: 10px;
}

.upload-btn {
  display: block;
  width: 100px;
  height: 100px;
  font-size: 16px;
  padding: 10px;
  background-color: #4CAF50;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin: auto;
}
    </style>
        </head>
        <body>
             <div class="center-box">
        <form action="/predict_okra" method="POST" enctype="multipart/form-data">
          <div class="upload-box">
            <label for="image-upload">Upload an image:</label>
            <input type="file" id="image-upload" name="image">
          </div>
          <button type="submit"> upload image</button>
        </form>
      </div>
        </body>
        </html>"""
    return HTMLResponse(content=content)

@app.post("/predict_okra")
async def predict(image: UploadFile = File(...)):
        array = ["Disesased okra leaf", "Healthy"]
        img = read_img(await image.read())
        img = img / 255
        # print(img)
        img = np.expand_dims(img, axis=0)
        prediction = model5.predict(img)
        if prediction[0] > 0.5:
            return {
                "Prediction": array[np.argmax(prediction[0]).tolist()],
                "Surity": str(np.max(prediction).tolist()*100) + " %"
            }
        else:
            return {
                "Prediction": "Healthy",
                "Surity": str(100 - np.max(prediction).tolist() * 100) + " %"
            }


@app.get("/upload_cabbage")
async def upload_page():
    content="""<!DOCTYPE html>
        <html>
        <head>
            <title>Image Upload Page</title>
            <style>
    .center-box {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.upload-box {
  border: 1px solid #ccc;
  padding: 10px;
  margin-bottom: 10px;
}

.upload-btn {
  display: block;
  width: 100px;
  height: 100px;
  font-size: 16px;
  padding: 10px;
  background-color: #4CAF50;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin: auto;
}
    </style>
        </head>
        <body>
             <div class="center-box">
        <form action="/predict_cabbage" method="POST" enctype="multipart/form-data">
          <div class="upload-box">
            <label for="image-upload">Upload an image:</label>
            <input type="file" id="image-upload" name="image">
          </div>
          <button type="submit"> upload image</button>
        </form>
      </div>
        </body>
        </html>"""
    return HTMLResponse(content=content)

@app.post("/predict_cabbage")
async def predict(image: UploadFile = File(...)):
        array = ["Backmoth", "Leafminer", "Mildew"]
        img = read_img(await image.read())
        img = img / 255
        # print(img)
        img = np.expand_dims(img, axis=0)
        prediction = model6.predict(img)
        return {
            "Disease": array[np.argmax(prediction[0]).tolist()],
            "Confidence": str(np.max(prediction).tolist()*100) + " %"
        }


'''if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)'''
