import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.
    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        save_best = kwargs.get('save_best', None)
        if isinstance(save_best, (list, tuple)):
            self._save_best_list = list(save_best)
            kwargs['save_best'] = self._save_best_list[0] if self._save_best_list else None
        else:
            self._save_best_list = None
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self._best_scores = {}

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=False,
            efficient_test=self.efficient_test)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt_multi(runner)

    def _save_ckpt_multi(self, runner):
        """Save multiple best checkpoints if configured."""
        if not self._save_best_list:
            self._save_ckpt(runner)
            return
        metrics = runner.log_buffer.output
        rule = getattr(self, 'rule', 'greater')
        for metric in self._save_best_list:
            if metric not in metrics:
                continue
            try:
                cur = float(metrics[metric])
            except Exception:
                continue
            best = self._best_scores.get(metric, None)
            if best is None:
                is_better = True
            else:
                is_better = cur > best if rule == 'greater' else cur < best
            if is_better:
                self._best_scores[metric] = cur
                if getattr(runner, 'rank', 0) == 0:
                    runner.save_checkpoint(
                        runner.work_dir,
                        filename_tmpl=f'best_{metric}.pth',
                        save_optimizer=True,
                        meta=runner.meta,
                        create_symlink=False)
                if runner.meta is not None:
                    runner.meta.setdefault('hook_msgs', {})
                    runner.meta['hook_msgs'][f'best_{metric}'] = cur


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.
    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        save_best = kwargs.get('save_best', None)
        if isinstance(save_best, (list, tuple)):
            self._save_best_list = list(save_best)
            kwargs['save_best'] = self._save_best_list[0] if self._save_best_list else None
        else:
            self._save_best_list = None
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self._best_scores = {}

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            efficient_test=self.efficient_test)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt_multi(runner)

    def _save_ckpt_multi(self, runner):
        """Save multiple best checkpoints if configured."""
        if not self._save_best_list:
            self._save_ckpt(runner)
            return
        metrics = runner.log_buffer.output
        rule = getattr(self, 'rule', 'greater')
        for metric in self._save_best_list:
            if metric not in metrics:
                continue
            try:
                cur = float(metrics[metric])
            except Exception:
                continue
            best = self._best_scores.get(metric, None)
            if best is None:
                is_better = True
            else:
                is_better = cur > best if rule == 'greater' else cur < best
            if is_better:
                self._best_scores[metric] = cur
                if runner.rank == 0:
                    runner.save_checkpoint(
                        runner.work_dir,
                        filename_tmpl=f'best_{metric}.pth',
                        save_optimizer=True,
                        meta=runner.meta,
                        create_symlink=False)
                if runner.meta is not None:
                    runner.meta.setdefault('hook_msgs', {})
                    runner.meta['hook_msgs'][f'best_{metric}'] = cur
