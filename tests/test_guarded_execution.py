"""Guard tests do not require PyDESeq2 or Streamlit."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock
import pytest
from omicsgpt.vnext_workflow import inspect_uploads, approve_review
from omicsgpt.guarded_execution import execute_approved_analysis, ExecutionNotAuthorized

ROOT = Path(__file__).resolve().parents[1]


def reviewed():
    b = inspect_uploads((ROOT/'bc_counts_transposed_condensed.csv').read_bytes(),
                        (ROOT/'bc_meta.csv').read_bytes(), sample_id_col='Sample',
                        group_col='condition', numerator='risk category: High',
                        reference='risk category: Average')
    return b, approve_review(b)


def test_valid_handoff_exact_policy_and_order():
    b, fp = reviewed()
    fake = Mock(return_value={'results': 'sentinel'})
    result = execute_approved_analysis(b, fp, b.config, engine=fake)
    assert result['results'] == 'sentinel'
    args, kw = fake.call_args
    assert args[0].shape == (187, 110)
    assert args[0].index.equals(args[1].index)
    assert kw['group_col'] == 'condition'
    assert kw['numerator'] == 'risk category: High'
    assert kw['reference'] == 'risk category: Average'
    assert kw['policy'].min_count == 10
    assert kw['policy'].min_fraction == .2
    assert kw['n_cpus'] == 1


@pytest.mark.parametrize('path,value', [
    (('analysis','min_count'), 11),
    (('analysis','min_fraction'), .3),
    (('analysis','dispersion_fit'), 'mean'),
    (('analysis','refit_cooks'), False),
    (('analysis','engine'), 'edgeR'),
    (('numerator',), 'risk category: Average'),
    (('reference',), 'risk category: High'),
    (('sample_order',), ['changed']),
    (('metadata_sha256',), '0'*64),
])
def test_changed_settings_block_before_engine(path, value):
    b, fp = reviewed()
    cfg = deepcopy(b.config)
    node = cfg
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    fake = Mock()
    with pytest.raises(ExecutionNotAuthorized):
        execute_approved_analysis(b, fp, cfg, engine=fake)
    fake.assert_not_called()


def test_missing_approval_blocks():
    b, _ = reviewed()
    fake = Mock()
    with pytest.raises(ExecutionNotAuthorized):
        execute_approved_analysis(b, '', b.config, engine=fake)
    fake.assert_not_called()


def test_modified_metadata_design_blocks():
    b, fp = reviewed()
    b.metadata.frame.loc[0, 'condition'] = 'new group'
    fake = Mock()
    with pytest.raises(ExecutionNotAuthorized):
        execute_approved_analysis(b, fp, b.config, engine=fake)
    fake.assert_not_called()


def test_engine_failure_propagates_without_fallback():
    b, fp = reviewed()
    fake = Mock(side_effect=RuntimeError('fitting failed'))
    with pytest.raises(RuntimeError, match='fitting failed'):
        execute_approved_analysis(b, fp, b.config, engine=fake)
    fake.assert_called_once()


def test_original_frames_not_mutated():
    b, fp = reviewed()
    c = b.counts.frame.copy(deep=True)
    m = b.metadata.frame.copy(deep=True)
    execute_approved_analysis(b, fp, b.config, engine=Mock(return_value={}))
    assert b.counts.frame.equals(c)
    assert b.metadata.frame.equals(m)
