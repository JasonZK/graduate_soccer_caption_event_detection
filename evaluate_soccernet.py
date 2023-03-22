import glob
import json
import os

from SoccerNet.Evaluation.ActionSpotting import evaluate, label2vector, predictions2vector

from tqdm import tqdm

PATH_DATASET = "D:/dataset/SoccerNet/SoccerNet_test_hq/"
PATH_PREDICTIONS = "D:/dataset/A_graduate_experiment/sub/socccernet_Json_result_2-23"

def evaluate_soccernet_by_soccernet(PATH_DATASET, PATH_PREDICTIONS):
    results = evaluate(SoccerNet_path=PATH_DATASET, Predictions_path=PATH_PREDICTIONS,
                       split="test", version=2, prediction_file="results_spotting.json", metric="loose")

    print("tight Average mAP: ", results["a_mAP"])
    print("tight Average mAP per class: ", results["a_mAP_per_class"])
    print("tight Average mAP visible: ", results["a_mAP_visible"])
    print("tight Average mAP visible per class: ", results["a_mAP_per_class_visible"])
    print("tight Average mAP unshown: ", results["a_mAP_unshown"])
    print("tight Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])


def my_getListGames(PATH_DATASET):
    list_games = []
    for root, matches, files in os.walk(PATH_DATASET):
        for match in matches:
            match_path = os.path.join(PATH_DATASET, match)
            for match_root, years, ffiles in os.walk(match_path):
                for year in years:
                    year_path = os.path.join(match_path, year)
                    for year_root, games, fffiles in os.walk(year_path):
                        for game in games:
                            game_path = os.path.join(year_path, game)
                            list_games.append(os.path.join(match, year, game))
                            # print(game_path)
                        break
                break
        break

    return list_games


def my_evaluate_soccernet(PATH_DATASET, PATH_PREDICTIONS, version=2, prediction_file="results_spotting.json", task="Substitution"):

    list_games = my_getListGames(PATH_DATASET)
    if version == 2:
        label_files = "Labels-v2.json"
        num_classes = 17
    else:
        label_files = "Labels.json"
        num_classes = 3

    gt_total = 0
    pre_total = 0
    right_total = 0
    for game in tqdm(list_games):
        labels = json.load(open(os.path.join(PATH_DATASET, game, label_files)))
        # label_half_1, label_half_2 = label2vector(labels, num_classes=num_classes, version=version)

        predictions = json.load(open(os.path.join(PATH_PREDICTIONS, game, prediction_file)))
        # predictions_half_1, predictions_half_2 = predictions2vector(predictions, num_classes=num_classes,version=version)

        gt_game_seconds = []
        for annotation in labels["annotations"]:
            if task in annotation["label"] :
                label_time = annotation["gameTime"]
                idx = annotation["gameTime"].split("-")[0].strip()
                goal_minutes = int(label_time[-5:-3])
                goal_seconds = int(label_time[-2::])
                gt_second = goal_seconds + 60 * goal_minutes
                gt_game_seconds.append(idx + "_" + str(gt_second))
        gt_total += len(gt_game_seconds)

        pre_game_seconds = []
        for prediction in predictions["predictions"]:
            if task in prediction["label"] and prediction["gameTime"][0]:
                predic_time = prediction["gameTime"].split("-")[1].strip()
                idx = prediction["gameTime"].split("-")[0].strip()
                goal_minutes = int(predic_time.split(":")[0])
                goal_seconds = int(predic_time.split(":")[1])
                pre_second = goal_seconds + 60 * goal_minutes
                pre_game_seconds.append(idx + "_" + str(pre_second))
        pre_total += len(pre_game_seconds)

        rest_gt_game = gt_game_seconds.copy()
        # rest_pre_game =
        for gt_second in gt_game_seconds:
            gt_idx = gt_second.split("_")[0]
            gt_second_num = int(gt_second.split("_")[1])
            for pre_second in pre_game_seconds:
                pre_idx = pre_second.split("_")[0]
                pre_second_num = int(pre_second.split("_")[1])
                if gt_idx == pre_idx and abs(pre_second_num - gt_second_num) <= 30:
                    rest_gt_game.remove(gt_second)
                    break
        right_total += (len(gt_game_seconds) - len(rest_gt_game))

    print("gt_total:{},  pre_total:{},  right_total:{}".format(gt_total, pre_total, right_total))



def test(path):
    files = os.listdir(path)

    results_dict = {}
    for file in files:
        print(file)


if __name__ == '__main__':
    print("6")
    # test(PATH_PREDICTIONS)
    # my_getListGames(PATH_DATASET)
    my_evaluate_soccernet(PATH_DATASET, PATH_PREDICTIONS, version=2)