import torch
import pytest
from sg_ttrans.data.sequence_buffer import SequenceBuffer
from sg_ttrans.data.dataset import NTHUDrowsinessDataset

def test_sequence_buffer_fifo():
    buf = SequenceBuffer(max_len=5)
    assert not buf.is_ready()
    for i in range(5):
        frame = torch.zeros(3, 224, 224)
        kin = torch.zeros(16)
        buf.push(frame, kin)
    assert buf.is_ready()
    frames, kins = buf.get_sequence()
    assert frames.shape == (5, 3, 224, 224)
    assert kins.shape == (5, 16)

    # Push 6th element, verify FIFO eviction
    new_frame = torch.ones(3, 224, 224)
    new_kin = torch.ones(16)
    buf.push(new_frame, new_kin)
    frames, kins = buf.get_sequence()
    assert torch.all(frames[-1] == 1.0)
    assert torch.all(kins[-1] == 1.0)
    assert len(buf) == 5

def test_nthu_dataset_structure_and_mock(tmp_path):
    # Create mock subject directory hierarchy:
    # tmp_path / "Subject01" / "Glasses_SlowBlink.mp4" (or frames)
    subj_dir = tmp_path / "Subject01"
    subj_dir.mkdir()
    
    dataset = NTHUDrowsinessDataset(
        root_dir=str(tmp_path),
        sequence_length=10,
        subjects=["Subject01"],
        mock_mode=True
    )
    assert len(dataset) > 0
    item = dataset[0]
    assert "frames" in item
    assert "kinematics" in item
    assert "label" in item
    assert item["frames"].shape == (10, 3, 224, 224)
    assert item["kinematics"].shape == (10, 16)
