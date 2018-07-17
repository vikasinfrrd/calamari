import argparse
from calamari_ocr.utils import split_all_ext, glob_all
from calamari_ocr.scripts.train import setup_train_args
from calamari_ocr.ocr import FileDataSet, Trainer
from calamari_ocr.ocr.augmentation.data_augmenter import SimpleDataAugmenter
from calamari_ocr.proto import CheckpointParams, DataPreprocessorParams, TextProcessorParams, \
    network_params_from_definition_string, NetworkParams

charset = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",   # Note positions
    "(", ")", " ",                                      # Compound/connected notes
    "F1", "F3", "F5", "F7",                             # F-Clef with position
    "C1", "C3", "C5", "C7",                             # C-Clef with position
]


def main():
    parser = argparse.ArgumentParser()
    setup_train_args(parser, omit=[])
    args = parser.parse_args()
    run(args)


def run(args):
    # check if loading a json file
    if len(args.files) == 1 and args.files[0].endswith("json"):
        import json
        with open(args.files[0], 'r') as f:
            json_args = json.load(f)
            for key, value in json_args.items():
                setattr(args, key, value)

    # parse whitelist
    whitelist = args.whitelist
    whitelist_files = glob_all(args.whitelist_files)
    for f in whitelist_files:
        with open(f) as txt:
            whitelist += list(txt.read())

    # Training dataset
    print("Resolving input files")
    input_image_files = glob_all(args.files)
    gt_txt_files = [split_all_ext(f)[0] + ".gt.txt" for f in input_image_files]

    if len(set(gt_txt_files)) != len(gt_txt_files):
        raise Exception("Some image are occurring more than once in the data set.")

    dataset = FileDataSet(input_image_files, gt_txt_files, skip_invalid=not args.no_skip_invalid_gt)
    print("Found {} files in the dataset".format(len(dataset)))

    # Validation dataset
    if args.validation:
        print("Resolving validation files")
        validation_image_files = glob_all(args.validation)
        val_txt_files = [split_all_ext(f)[0] + ".gt.txt" for f in validation_image_files]

        if len(set(val_txt_files)) != len(val_txt_files):
            raise Exception("Some validation images are occurring more than once in the data set.")

        validation_dataset = FileDataSet(validation_image_files, val_txt_files,
                                         skip_invalid=not args.no_skip_invalid_gt)
        print("Found {} files in the validation dataset".format(len(validation_dataset)))
    else:
        validation_dataset = None

    params = CheckpointParams()

    params.max_iters = args.max_iters
    params.stats_size = args.stats_size
    params.batch_size = args.batch_size
    params.checkpoint_frequency = args.checkpoint_frequency
    params.output_dir = args.output_dir
    params.output_model_prefix = args.output_model_prefix
    params.display = args.display
    params.skip_invalid_gt = not args.no_skip_invalid_gt
    params.processes = args.num_threads

    params.early_stopping_frequency = args.early_stopping_frequency if args.early_stopping_frequency >= 0 else args.checkpoint_frequency
    params.early_stopping_nbest = args.early_stopping_nbest
    params.early_stopping_best_model_prefix = args.early_stopping_best_model_prefix
    params.early_stopping_best_model_output_dir = \
        args.early_stopping_best_model_output_dir if args.early_stopping_best_model_output_dir else args.output_dir

    params.model.data_preprocessor.type = DataPreprocessorParams.MULTI_NORMALIZER
    params.model.data_preprocessor.children.add().type = DataPreprocessorParams.RANGE_NORMALIZER
    scale_to_h_params = params.model.data_preprocessor.children.add()
    scale_to_h_params.type = DataPreprocessorParams.SCALE_TO_HEIGHT
    scale_to_h_params.line_height = args.line_height
    final_prep_params = params.model.data_preprocessor.children.add()
    final_prep_params.type = DataPreprocessorParams.FINAL_PREPARATION
    final_prep_params.pad = args.pad

    # Text pre processing (reading)
    params.model.text_preprocessor.type = TextProcessorParams.MULTI_NORMALIZER
    params.model.text_preprocessor.children.add().type = TextProcessorParams.STRIP_NORMALIZER
    str_to_char_list = params.model.text_preprocessor.children.add()
    str_to_char_list.type = TextProcessorParams.STR_TO_CHAR_LIST
    str_to_char_list.characters[:] = sorted(charset, key=lambda a: -len(a))

    # Text post processing (prediction)
    params.model.text_postprocessor.type = TextProcessorParams.STRIP_NORMALIZER

    if args.seed > 0:
        params.model.network.backend.random_seed = args.seed

    params.model.line_height = args.line_height

    network_params_from_definition_string(args.network, params.model.network)
    params.model.network.clipping_mode = NetworkParams.ClippingMode.Value("CLIP_" + args.gradient_clipping_mode.upper())
    params.model.network.clipping_constant = args.gradient_clipping_const
    params.model.network.backend.fuzzy_ctc_library_path = args.fuzzy_ctc_library_path
    params.model.network.backend.num_inter_threads = args.num_inter_threads
    params.model.network.backend.num_intra_threads = args.num_intra_threads

    # create the actual trainer
    trainer = Trainer(params,
                      dataset,
                      validation_dataset=validation_dataset,
                      data_augmenter=SimpleDataAugmenter(),
                      n_augmentations=args.n_augmentations,
                      weights=args.weights,
                      )
    trainer.train(progress_bar=not args.no_progress_bars)


if __name__ == '__main__':
    main()