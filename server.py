import sys
import time

import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


def holder_table(x1, x2):
    time.sleep(0.2)
    X = (x1, x2)
    return -abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))


@app.route('/run_objective', methods=['GET'])
def run_objective():
    input_parameters = request.args.to_dict()
    try:
        parameters = {key: float(value) for key, value in input_parameters.items()}
    except ValueError:
        res = jsonify({"data": "Bad request: All parameters must be numeric"})
        res.status_code = 400
        return res

    return jsonify({"data": holder_table(**parameters)})


if __name__ == '__main__':
    app.run(
        debug=True,
        port=int(sys.argv[1]),
        threaded=False
    )
