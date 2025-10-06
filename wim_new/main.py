from WiM_inference import run_wim_rl

if __name__ == '__main__':
    
    model_id = "shubvhamgore18218/WiM_llama_full_dataset"  # or any compatible model identifier
    model_id_rl = 'shubvhamgore18218/WiM_llama_full_dataset'
    # input_document = "Your long document text goes here..."
    query = "Who is silky Epeira?"
    
    # Set parameters for margin generation and RL training
    final_answer, positive_margins = run_wim_rl(
        model_id=model_id,
        model_id_rl=model_id_rl,
        input_document='examples/babilong_8k.json',
        query=query,
        use_rl_generator=True,      # Use the RL-enhanced margin generator
        train_rl_generator=False,   # Set True if you want to train the RL generator
        num_episodes=1,            
        # output_model_dir='./output'
    )
    
    print("Final Answer:")
    print(final_answer)
    print("\nPositive Margins:")
    for margin in positive_margins:
        print(margin)