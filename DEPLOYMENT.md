# Deployment Guide: HuggingFace Spaces

## Quick Deploy to HuggingFace Spaces

### Option 1: Direct Upload (Easiest)

1. **Create a HuggingFace Space**
   - Go to https://huggingface.co/new-space
   - Choose **Gradio** as SDK
   - Choose a name (e.g., `esperanto-learning`)
   - Select **Public** visibility

2. **Upload Files**
   - Upload `app.py`
   - Upload `requirements.txt`
   - Upload entire `data/` folder
   - Upload `figures/` folder (optional, for fallback images)

3. **Wait for Build**
   - HuggingFace automatically builds and deploys
   - Your space will be live at `https://huggingface.co/spaces/your-username/esperanto-learning`

### Option 2: Git Push (Recommended)

```bash
# 1. Create space on HuggingFace.co and get the git URL

# 2. Add HF remote
git remote add hf https://huggingface.co/spaces/your-username/esperanto-learning

# 3. Create deployment branch
git checkout -b deploy

# 4. Copy necessary files to root (if not already there)
# app.py, requirements.txt, data/, figures/

# 5. Push to HuggingFace
git push hf deploy:main
```

### Option 3: Automated with GitHub Actions

Create `.github/workflows/deploy-hf.yml`:

```yaml
name: Deploy to HuggingFace Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Push to HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config --global user.email "action@github.com"
          git config --global user.name "GitHub Action"
          git remote add hf https://user:$HF_TOKEN@huggingface.co/spaces/your-username/esperanto-learning
          git push hf main:main
```

## Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
python app.py

# Open browser to http://localhost:7860
```

## Space Configuration

Create a `README.md` in your HuggingFace Space:

```markdown
---
title: Esperanto Learning Dataset Explorer
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Esperanto Learning with ChatGPT

Interactive explorer for analyzing 397 ChatGPT conversations from Esperanto learners.

See full documentation: https://github.com/lukketsvane/esperanto-db
```

## Performance Optimization

For large datasets, consider:

1. **Data Sampling**: Load subset for faster initial load
   ```python
   df = pd.read_csv("data/conversations_final.csv").sample(n=100)
   ```

2. **Lazy Loading**: Load visualizations on-demand
   ```python
   @gr.cache
   def create_plot():
       # Expensive computation
   ```

3. **Persistent Storage**: Use HuggingFace Datasets API
   ```python
   from datasets import load_dataset
   dataset = load_dataset("your-username/esperanto-conversations")
   ```

## Troubleshooting

### Build Fails

- Check `requirements.txt` versions are compatible
- Ensure all data files are uploaded
- Check logs in HuggingFace Space settings

### Slow Loading

- Reduce initial visualizations
- Use Gradio's `lazy` loading for tabs
- Consider data preprocessing

### Memory Issues

- HuggingFace Spaces have memory limits
- Optimize data structures (use `pd.Categorical` for strings)
- Consider upgrading to paid tier for more resources

## Custom Domain

1. Go to Space Settings
2. Add custom domain (requires HF Pro)
3. Update DNS records

## Analytics

Enable analytics in Space settings to track:
- Page views
- User interactions
- Geographic distribution

## Updating

To update deployed space:

```bash
# Make changes to app.py or data/
git commit -m "Update visualizations"
git push hf deploy:main
```

Changes deploy automatically within minutes.

## Embedding

Embed your space in websites:

```html
<iframe
  src="https://huggingface.co/spaces/your-username/esperanto-learning"
  frameborder="0"
  width="850"
  height="450"
></iframe>
```

## Best Practices

1. **Add Description**: Write clear README for your Space
2. **Add Tags**: Use relevant tags (language-learning, dataset, chatgpt)
3. **Pin Space**: Pin to your profile for visibility
4. **Share**: Tweet, LinkedIn, Reddit (r/MachineLearning, r/LanguageLearning)
5. **Monitor**: Check analytics and user feedback

## Resources

- HuggingFace Spaces Docs: https://huggingface.co/docs/hub/spaces
- Gradio Docs: https://gradio.app/docs/
- Example Spaces: https://huggingface.co/spaces

---

For issues, see: https://github.com/lukketsvane/esperanto-db/issues
