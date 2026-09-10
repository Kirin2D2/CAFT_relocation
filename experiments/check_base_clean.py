# Is the kernel model (post-LoRA-unload) identical to a fresh disk load?
# Compare eval under the SAME harness, then compare actual weights.
import torch as t, gc
from transformers import AutoModelForCausalLM
kernel_met, _ = eval_split(model, GenderDataset(train_ambiguous_frac=1.0, test_ambiguous_frac=0.0).test)
print(f"kernel model  OOD acc_2way = {kernel_met['acc_2way']:.4f}  ties={kernel_met['n_ties']}")
fresh = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Base", device_map="cuda:0", dtype=t.bfloat16)
fresh.eval()
fresh_met, _ = eval_split(fresh, GenderDataset(train_ambiguous_frac=1.0, test_ambiguous_frac=0.0).test)
print(f"fresh model   OOD acc_2way = {fresh_met['acc_2way']:.4f}  ties={fresh_met['n_ties']}")
diffs = 0
for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh.named_parameters()):
    assert n1 == n2
    if not t.equal(p1, p2):
        diffs += 1
        if diffs <= 5: print("  WEIGHT DIFF:", n1)
print(f"weight tensors differing: {diffs} / {sum(1 for _ in model.parameters())}")
del fresh; gc.collect(); t.cuda.empty_cache()
print("CHECK_DONE")
