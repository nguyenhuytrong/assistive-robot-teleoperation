
from typing import List, Optional, Tuple
import torch

def post_process_semantic_segmentation1(outputs, target_sizes: Optional[List[Tuple[int, int]]] = None) -> "torch.Tensor":
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.class_queries_logits.cuda()  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits.cuda()  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        # masks_classes = masks_classes.cuda()
        # masks_probs = masks_probs.cuda()
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

def ade_palette(option=1):
    array = [(250, 250, 250) for i in range(151)]
    if option == 1:
        array[0] = (0, 0, 0)
        array[3] = (0, 250, 0)
        array[12] = (0, 0, 250)
        array[138] = (224, 5, 225)
    elif option == 2:
        array[0] = (0, 0, 0)
        array[3] = (0, 0, 0)
        array[12] = (0, 0, 250)
        array[138] = (0, 0, 250)
    # Always return, regardless of option value
    return array
    
    