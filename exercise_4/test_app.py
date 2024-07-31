import pytest
from flask import Flask
from flask.testing import FlaskClient
import cv2
import numpy as np
import io

from app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_predict(client: FlaskClient):
    image = np.zeros((28, 28), dtype=np.uint8)
    _, image_encoded = cv2.imencode(".png", image)
    image_bytes = image_encoded.tobytes()

    data = {"image": (io.BytesIO(image_bytes), "test.png")}

    response = client.post("/predict", content_type="multipart/form-data", data=data)
    assert response.status_code == 200
    json_data = response.get_json()
    assert "prediction" in json_data
