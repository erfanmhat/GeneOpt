import sys
import time

from flask import Flask, jsonify, request

app = Flask(__name__)


def objective_func(a, b, c):
    time.sleep(0.2)
    return (a - 5) ** 2 + (b - 20) ** 2 + (c - 33.33) ** 4


@app.route('/run_objective', methods=['GET'])
def run_objective():
    input_parameters = request.args.to_dict()
    try:
        parameters = {key: float(value) for key, value in input_parameters.items()}
    except ValueError:
        res = jsonify({"data": "Bad request: All parameters must be numeric"})
        res.status_code = 400
        return res

    return jsonify({"data": objective_func(**parameters)})


if __name__ == '__main__':
    app.run(
        debug=True,
        port=int(sys.argv[1]),
        threaded=False
    )
