import numpy as np
import pandas as pd
import joblib
import argparse  # argument via terminal

# load model and encoder
model = joblib.load('quality_prediction_linear_model_with_oss_categories.joblib')
encoder = joblib.load('oss_category_encoder_with_oss_categories.joblib')

def translate_ranking(ranking):
    if ranking >= 4.5:
        return '++', ranking
    elif ranking >= 3.5:
        return '+', ranking
    elif ranking >= 2.5:
        return 'o', ranking
    elif ranking >= 1.5:
        return '-', ranking
    else:
        return '--', ranking


def predict_future_quality(current_oss_category, current_rankings):
    future_oss_categories = ['Attractive', 'Fluctuating', 'Stagnant', 'Terminal']
    maintainability = ['analysability', 'changeability', 'testability']
    predictions = {}
    for future_category in future_oss_categories:
        input_df = pd.DataFrame({'oss_category': [current_oss_category], 'prev_oss_category': [future_category]})
        category_encoded = encoder.transform(input_df)
        features = np.hstack((category_encoded, np.array(current_rankings).reshape(1, -1)))
        future_rankings_predicted = model.predict(features)
        # min 1 max 5
        future_rankings_predicted = np.clip(future_rankings_predicted, 1, 5)
        
        #predictions[future_category] = future_rankings_predicted[0]
        
        translated_rankings = [translate_ranking(ranking) for ranking in future_rankings_predicted[0]]
        predictions[future_category] = dict(zip(maintainability, translated_rankings))

    return predictions

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description='Predict maintainability rankings for OSS categories.')
    parser.add_argument('current_oss_category', type=str, help='Current OSS category')
    parser.add_argument('rankings', nargs=3, type=int, help='Current maintainability rankings (analysability, changeability, and testability) as integers')
    args = parser.parse_args()


    if args.current_oss_category not in encoder.categories_[0]:
        print("Unknown OSS category...")
        
    else:
        
        # predict
        predictions = predict_future_quality(args.current_oss_category, args.rankings)
        
        for future_category, metrics in predictions.items():
            print(f"{future_category}:")
            for metric, (ranking_symbol, ranking_value) in metrics.items():
                print(f"  {metric}: {ranking_symbol} ({ranking_value:.2f})")
            print("===========================")



