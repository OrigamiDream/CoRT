import grpc
import logging
import argparse
import numpy as np
import tensorflow as tf

from utils import utils
from flask import Flask, request, jsonify, redirect, render_template
from cort.preprocessing import normalize_texts, REMOVABLE_SPECIAL_CHAR_REPLACEMENTS, SPECIAL_CHAR_REPLACEMENTS
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from tensorflow_serving.apis.predict_pb2 import PredictRequest


def request_prediction(sentence, tokenizer, replacements, channel, args):
    orig = sentence
    # normalize texts
    sentence = normalize_texts(sentence, remove_specials=False, remove_last_period=False)
    sentence = sentence.lower()

    # tokenize texts
    tokenized = tokenizer([sentence],
                          padding='max_length',
                          truncation=True,
                          return_attention_mask=False,
                          return_token_type_ids=False)
    input_ids = np.array(tokenized['input_ids'], dtype=np.int32)

    # build gRPC predict request
    predict_req = PredictRequest()
    predict_req.model_spec.name = args.model_spec_name
    predict_req.model_spec.signature_name = args.signature_name
    predict_req.inputs['input_ids'].CopyFrom(tf.make_tensor_proto(input_ids, dtype=tf.int32))

    stub = PredictionServiceStub(channel)
    response = stub.Predict(predict_req)

    # reshape response
    probs = np.array(response.outputs['probs'].float_val).reshape((1, -1))
    correlations = np.array(response.outputs['correlations'].float_val).reshape((1, -1))

    # shear correlation vector
    sequence_length = np.sum(input_ids != tokenizer.pad_token_id)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, 1:sequence_length - 1])  # remove [CLS], [SEP], [PAD]
    correlations = correlations[0, 1:sequence_length - 1]
    correlations = (correlations - np.min(correlations)) / (np.max(correlations) - np.min(correlations))  # normalize

    # compose correlation vector to tokens for graphs
    composed_tokens = utils.compose_correlation_to_tokens(correlations, tokens, orig, replacements)
    prediction = int(np.argmax(probs[0]))
    prediction_prob = float(probs[0][prediction])
    probs = [float(prob) for prob in probs[0]]
    return {
        'prediction': prediction,
        'prediction_prob': prediction_prob,
        'probs': probs,
        'composed_tokens': composed_tokens
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--host', default='0.0.0.0',
                        help='Listening address for Flask server')
    parser.add_argument('--port', default=8080,
                        help='Number of port for Flask server')
    parser.add_argument('--grpc_server', default='localhost:8500',
                        help='Address to TFServing gRPC API endpoint.')
    parser.add_argument('--model_name', default='korscibert',
                        help='Name of pre-trained models. (One of korscibert, korscielectra, huggingface models)')
    parser.add_argument('--model_spec_name', default='cort',
                        help='Name of model spec.')
    parser.add_argument('--signature_name', default='serving_default',
                        help='Name of signature of SavedModel')
    parser.add_argument('--debug', default=False,
                        help='Debug mode')

    # Configurable pre-defined variables
    parser.add_argument('--korscibert_vocab', default='./cort/pretrained/korscibert/vocab_kisti.txt')
    parser.add_argument('--korscibert_ckpt', default='./cort/pretrained/korscibert/model.ckpt-262500')
    parser.add_argument('--korscielectra_vocab', default='./cort/pretrained/korscielectra/data/vocab.txt')
    parser.add_argument('--korscielectra_ckpt', default='./cort/pretrained/korscielectra/data/models/korsci_base')

    args = parser.parse_args()
    tokenizer = utils.create_tokenizer_from_config(args)
    if hasattr(tokenizer, 'disable_progressbar'):  # disable tqdm on korscibert tokenizer
        tokenizer.disable_progressbar = True

    replacements = REMOVABLE_SPECIAL_CHAR_REPLACEMENTS + SPECIAL_CHAR_REPLACEMENTS
    replacements = [(before, tokenizer.tokenize(after)) for before, after in replacements]

    channel = grpc.insecure_channel(args.grpc_server)
    app = Flask(__name__,
                static_url_path='/static',
                static_folder='./public/static',
                template_folder='./public/template')

    @app.errorhandler(404)
    def handle_not_found(_):
        return redirect('/site')

    @app.errorhandler(Exception)
    def handle_exception(e):
        return jsonify({
            'error': str(e)
        })

    @app.route('/site', methods=['GET'])
    def site():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        body = request.get_json()
        try:
            outputs = request_prediction(body['sentence'], tokenizer, replacements, channel, args)
        except grpc.RpcError as e:
            logging.error(e)
            return jsonify({
                'error': 'grpc_backend_unavailable'
            })
        else:
            return jsonify({
                'error': None,
                'prediction': outputs['prediction'],
                'prediction_prob': outputs['prediction_prob'],
                'probs': outputs['probs'],
                'composed_tokens': outputs['composed_tokens']
            })

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
