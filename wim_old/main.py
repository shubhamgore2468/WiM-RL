from WiM_inference import run_wim_rl

if __name__ == '__main__':
    # Define  model and input parameters
    model_id = "shubvhamgore18218/WiM_llama"  
    query = "What is the key idea of the document?"
    
    final_answer, positive_margins = run_wim_rl(
        model_id=model_id,
        input_document='examples/babilong_8k.json',
        query=query,
        use_rl_generator=True,      
        train_rl_generator=True,   
        num_episodes=3            
    )
    
    print("Final Answer:")
    print(final_answer)
    print("\nPositive Margins:")
    for margin in positive_margins:
        print(margin)