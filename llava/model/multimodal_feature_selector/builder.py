from transformers import Blip2Processor, Blip2Model, Blip2Config
import torch
import torch.nn as nn

class ImageQFormer(nn.Module):
    def __init__(self):
        super().__init__()
        print("build model with qformerv2")
        self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.vision_model = None
        self.model.language_model = None
        self.model.language_projection = None

    def forward(self, image_embeds, **kwargs):
        # Forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device) #is this correct?

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
            return_dict=True,
        )
        query_output = query_outputs[0]
        #or
        query_output = query_outputs.last_hidden_state
        return query_outputs
        #TODO discard irrelevant features and upsample relevant features


def build_vision_feature_selector(config, **kwargs):
    return ImageQFormer()


