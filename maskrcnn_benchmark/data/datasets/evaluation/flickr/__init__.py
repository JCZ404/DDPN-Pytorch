from .flickr_eval import eval_recall


def flick_evaluation(dataset, predictions, image_ids, output_folder=None):
    return eval_recall(dataset, predictions, image_ids, output_folder=None)