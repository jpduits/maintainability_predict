import numpy as np
import pandas as pd
import joblib
import argparse  # argument via terminal

model = joblib.load('quality_prediction_linear_model_with_oss_categories.joblib')
encoder = joblib.load('oss_category_encoder_with_oss_categories.joblib')

def translate_to_sig_ranking(ranking):
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


def predict_future_rankings_based_on_current_state(current_oss_category, current_rankings):
    
    possible_future_oss_categories = ['Attractive', 'Fluctuating', 'Stagnant', 'Terminal']
    maintainability_characteristics = ['Analysability', 'Changeability', 'Testability']

    predictions = {}

    for future_oss_category in possible_future_oss_categories:
        # dataframe
        input_df = pd.DataFrame({
            'oss_category': [future_oss_category],  # OSS cateogry to predict
            'prev_oss_category': [current_oss_category]  # Current OSS cateogory
        })
        categories_encoded = encoder.transform(input_df)
        
        # merge with rankings
        features = np.hstack((categories_encoded, np.array(current_rankings).reshape(1, -1)))
        
        # use the trained model to predict with the features
        future_rankings_predicted = model.predict(features)
        
        # max min ranking value
        future_rankings_predicted = np.clip(future_rankings_predicted, 1, 5)
        
        # translate to SIG ranking (++, +, o, -, --)
        readable_rankings = [translate_to_sig_ranking(ranking) for ranking in future_rankings_predicted[0]]
        
        # store rankings in the predictions array
        predictions[future_oss_category] = dict(zip(maintainability_characteristics, readable_rankings))

    return predictions


if __name__ == "__main__":
    # arguments from terminal
    parser = argparse.ArgumentParser(description='Predict maintainability rankings for OSS categories.')
    parser.add_argument('current_oss_category', type=str, help='Current OSS category')
    parser.add_argument('rankings', nargs=3, type=float, help='Current maintainability rankings (analysability, changeability, and testability)')
    args = parser.parse_args()


    # valid input
    if args.current_oss_category not in encoder.categories_[0]:
        print("Unknown OSS category!")
        
    else:
        
        print(f"\nCurrent OSS Category: {args.current_oss_category}\n")
        print("Current rankings:")
        for aspect, ranking in zip(['Analysability', 'Changeability', 'Testability'], args.rankings):
            ranking_symbol, _ = translate_to_sig_ranking(ranking)
            print(f"{aspect}:\t{ranking_symbol}\t({ranking:.2f})")
        print("\n==============================\n")
        
        
        # predict
        predictions = predict_future_rankings_based_on_current_state(args.current_oss_category, args.rankings)
        
        for future_category, metrics in predictions.items():
            print(f"If next period is {future_category},\nthe predicted rankings are:\n")
            for metric, (ranking_symbol, ranking_value) in metrics.items():
                print(f"{metric}:\t{ranking_symbol}\t({ranking_value:.2f})")
            print("\n==============================\n")



