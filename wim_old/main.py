from WiM_inference import run_wim_rl

if __name__ == '__main__':
    # Define your model and input parameters
    model_id = "shubvhamgore18218/WiM_llama"  # or any compatible model identifier
    query = "What is the key idea of the document?"
    
    # Set parameters for margin generation and RL training
    final_answer, positive_margins = run_wim_rl(
        model_id=model_id,
        input_document='examples/babilong_8k.json',
        query=query,
        use_rl_generator=True,      # Use the RL-enhanced margin generator
        train_rl_generator=True,   # Set True if you want to train the RL generator
        num_episodes=3             # Number of training episodes (only used if training)
    )
    
    print("Final Answer:")
    print(final_answer)
    print("\nPositive Margins:")
    for margin in positive_margins:
        print(margin)