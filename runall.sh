#!/bin/bash

# attack

# python top200exp_nontarget.py nontarget_objective_fn_rcnn_cls
# python top200exp_nontarget.py nontarget_objective_fn_rcnn_reg
# python top200exp_nontarget.py nontarget_objective_fn_rcnn_cls_reg

# python top200exp_nontarget.py nontarget_objective_fn_rpn_cls
# python top200exp_nontarget.py nontarget_objective_fn_rpn_reg
# python top200exp_nontarget.py nontarget_objective_fn_rpn_cls_reg

# python top200exp_nontarget.py nontarget_objective_fn_rpn_rcnn_cls
# python top200exp_nontarget.py nontarget_objective_fn_rpn_rcnn_reg
# python top200exp_nontarget.py nontarget_objective_fn_rpn_rcnn_cls_reg

# python top200exp_nontarget.py nontarget_objective_fn_rcnn_cnn_cls
# python top200exp_nontarget.py nontarget_objective_fn_rcnn_cnn_reg
# python top200exp_nontarget.py nontarget_objective_fn_rcnn_cnn_cls_reg

# detect in the adversarial examples

# python infer_adv_examples.py nontarget_objective_fn_rcnn_cls
# python infer_adv_examples.py nontarget_objective_fn_rcnn_reg
# python infer_adv_examples.py nontarget_objective_fn_rcnn_cls_reg

# python infer_adv_examples.py nontarget_objective_fn_rpn_cls
# python infer_adv_examples.py nontarget_objective_fn_rpn_reg
# python infer_adv_examples.py nontarget_objective_fn_rpn_cls_reg

# python infer_adv_examples.py nontarget_objective_fn_rpn_rcnn_cls
# python infer_adv_examples.py nontarget_objective_fn_rpn_rcnn_reg
# python infer_adv_examples.py nontarget_objective_fn_rpn_rcnn_cls_reg

# python infer_adv_examples.py nontarget_objective_fn_rcnn_cnn_cls
# python infer_adv_examples.py nontarget_objective_fn_rcnn_cnn_reg
# python infer_adv_examples.py nontarget_objective_fn_rcnn_cnn_cls_reg

# eval the performance on adversarial examples

# echo nontarget_objective_fn_rcnn_cls
# python eval_adv_examples.py nontarget_objective_fn_rcnn_cls
# echo nontarget_objective_fn_rcnn_reg
# python eval_adv_examples.py nontarget_objective_fn_rcnn_reg
# echo nontarget_objective_fn_rcnn_cls_reg
# python eval_adv_examples.py nontarget_objective_fn_rcnn_cls_reg

# echo nontarget_objective_fn_rpn_cls
# python eval_adv_examples.py nontarget_objective_fn_rpn_cls
# echo nontarget_objective_fn_rpn_reg
# python eval_adv_examples.py nontarget_objective_fn_rpn_reg
# echo nontarget_objective_fn_rpn_cls_reg
# python eval_adv_examples.py nontarget_objective_fn_rpn_cls_reg

# echo nontarget_objective_fn_rpn_rcnn_cls
# python eval_adv_examples.py nontarget_objective_fn_rpn_rcnn_cls
# echo nontarget_objective_fn_rpn_rcnn_reg
# python eval_adv_examples.py nontarget_objective_fn_rpn_rcnn_reg
# echo nontarget_objective_fn_rpn_rcnn_cls_reg
# python eval_adv_examples.py nontarget_objective_fn_rpn_rcnn_cls_reg

# echo nontarget_objective_fn_rcnn_cnn_cls
# python eval_adv_examples.py nontarget_objective_fn_rcnn_cnn_cls
# echo nontarget_objective_fn_rcnn_cnn_reg
# python eval_adv_examples.py nontarget_objective_fn_rcnn_cnn_reg
# echo nontarget_objective_fn_rcnn_cnn_cls_reg
# python eval_adv_examples.py nontarget_objective_fn_rcnn_cnn_cls_reg