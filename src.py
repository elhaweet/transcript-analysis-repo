import os
import re
import json
import time
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CallCenterProcessor:
    def __init__(self):
        """Initialize the Call Center Processor with Gemini API"""
        # Load environment variables
        load_dotenv()
        
        # Configure the API key
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Rate limiting
        self.request_delay = 1  # seconds between requests
        self.max_retries = 3
        
        
    def process_dataframe(self, df: pd.DataFrame, output_file: str, batch_size: int = 10):
        """Process a DataFrame directly instead of reading CSV inside"""
        try:
            logger.info(f"Processing {len(df)} records")

            all_results = []
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")

                batch_results = []
                for idx, row in batch.iterrows():
                    logger.info(f"Processing call {idx + 1}/{len(df)}: {row.get('Call ID', 'N/A')}")
                    analysis = self.process_single_call(row)
                    flattened = self.flatten_analysis(analysis)
                    batch_results.append(flattened)
                    time.sleep(self.request_delay)

                all_results.extend(batch_results)

                if i % (batch_size * 5) == 0:
                    self.save_intermediate_results(df, all_results, output_file, i + batch_size)

            results_df = pd.DataFrame(all_results)
            final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
            final_df.to_csv(output_file, index=False)

            logger.info(f"Processing complete. Results saved to: {output_file}")
            self.generate_summary_report(final_df, output_file.replace('.csv', '_summary.txt'))

        except Exception as e:
            logger.error(f"Error processing DataFrame: {e}")
            raise
   
        
    def preprocess_transcript(self, text: str) -> str:
        """Clean and preprocess transcript text for LLM analysis"""
        if not text or pd.isna(text):
            return ""
            
        # Remove transcription artifacts
        text = re.sub(r'\[inaudible\]|\[unclear\]|\[noise\]|\[pause\]|\[silence\]|\[background noise\]', '', text)
        text = re.sub(r'\[music\]|\[hold music\]|\[beep\]', '', text)
        
        # Remove timestamps
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|\(\d{2}:\d{2}:\d{2}\)', '', text)
        
        # Standardize speaker labels
        speaker_patterns = [
            (r'Agent\s*\d*\s*:', 'Agent:'),
            (r'Representative\s*\d*\s*:', 'Agent:'),
            (r'Rep\s*\d*\s*:', 'Agent:'),
            (r'Customer\s*\d*\s*:', 'Customer:'),
            (r'Caller\s*\d*\s*:', 'Customer:'),
            (r'Client\s*\d*\s*:', 'Customer:'),
        ]
        
        for pattern, replacement in speaker_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Reduce filler words
        text = re.sub(r'\b(um|uh|er|ah)\b(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(like|you know|so|well)\b(\s+\1\b){2,}', r'\1', text, flags=re.IGNORECASE)
        
        # Normalize text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        
        # Standardize numbers and dates
        text = re.sub(r'\b(\d{3})[-.]\s*(\d{3})[-.]\s*(\d{4})\b', r'\1-\2-\3', text)
        text = re.sub(r'\$\s*(\d+)', r'$\1', text)
        
        return text.strip()
    
    def preprocess_summary(self, summary: str) -> str:
        """Clean and structure summary for better LLM processing"""
        if not summary or pd.isna(summary):
            return ""
            
        # Remove boilerplate
        summary = re.sub(r'Call Summary:?|Summary:?', '', summary, flags=re.IGNORECASE)
        
        # Clean and normalize
        summary = re.sub(r'\s+', ' ', summary)
        summary = summary.strip()
        
        return summary
    
    def create_analysis_prompt(self, call_data: Dict) -> str:
        """Create a comprehensive analysis prompt for Gemini"""
        
        transcript = call_data.get('processed_transcript', '')
        summary = call_data.get('processed_summary', '')
        duration = call_data.get('duration', 'N/A')
        ended_reason = call_data.get('ended_reason', 'N/A')
        
        prompt = f"""
Analyze this call center interaction and provide detailed insights in JSON format.

CALL METADATA:
- Call ID: {call_data.get('call_id', 'N/A')}
- Duration: {duration}
- Ended Reason: {ended_reason}
- Customer: {call_data.get('customer_name', 'N/A')}

TRANSCRIPT:
{transcript}

SUMMARY:
{summary}

Please analyze this call and return a JSON object with the following structure:

{{
    "sentiment_analysis": {{
        "overall_sentiment": "positive/negative/neutral",
        "overall_sentiment_score": 0.0,
        "customer_sentiment": "positive/negative/neutral",
        "customer_sentiment_score": 0.0,
        "agent_sentiment": "positive/negative/neutral",
        "agent_sentiment_score": 0.0,
        "sentiment_journey": "improved/declined/stable",
        "emotional_indicators": ["calm", "frustrated", "satisfied", "angry", "confused"]
    }},
    "intent_classification": {{
        "primary_intent": "inquiry/complaint/request/support/sales",
        "secondary_intent": "inquiry/complaint/request/support/sales/none",
        "intent_confidence": 0.0,
        "specific_intent": "billing_inquiry/technical_support/product_question/service_complaint/cancellation_request/upgrade_request/general_inquiry"
    }},
    "entity_extraction": {{
        "customer_entities": {{
            "account_numbers": [],
            "phone_numbers": [],
            "email_addresses": [],
            "order_ids": [],
            "reference_numbers": []
        }},
        "business_entities": {{
            "products_mentioned": [],
            "services_mentioned": [],
            "competitors_mentioned": [],
            "departments_mentioned": [],
            "policies_mentioned": []
        }},
        "issue_entities": {{
            "problem_categories": ["billing", "technical", "product", "service", "account"],
            "urgency_level": "low/medium/high/critical",
            "resolution_status": "resolved/pending/escalated/unresolved"
        }}
    }},
    "conversation_analysis": {{
        "call_structure_score": 0.0,
        "agent_professionalism": 0.0,
        "customer_satisfaction_predicted": 0.0,
        "communication_clarity": 0.0,
        "problem_resolution_effectiveness": 0.0,
        "call_flow_quality": "excellent/good/fair/poor"
    }},
    "quality_metrics": {{
        "agent_performance": {{
            "politeness_score": 0.0,
            "empathy_score": 0.0,
            "technical_accuracy": 0.0,
            "response_time_quality": "excellent/good/fair/poor",
            "follow_up_quality": "excellent/good/fair/poor"
        }},
        "customer_experience": {{
            "satisfaction_indicators": ["positive_feedback", "issue_resolved", "polite_interaction"],
            "frustration_indicators": ["long_wait", "multiple_transfers", "unresolved_issue"],
            "loyalty_impact": "positive/negative/neutral"
        }}
    }},
    "business_insights": {{
        "churn_risk": "low/medium/high",
        "upsell_opportunity": "low/medium/high",
        "training_needs": ["communication", "technical", "product_knowledge", "empathy"],
        "process_improvement_areas": ["call_routing", "knowledge_base", "escalation_process"],
        "cost_efficiency": "high/medium/low"
    }},
    "topic_analysis": {{
        "main_topics": ["billing", "technical_support", "product_inquiry", "service_complaint"],
        "subtopics": [],
        "topic_complexity": "low/medium/high",
        "resolution_complexity": "low/medium/high"
    }},
    "outcome_analysis": {{
        "call_success": true,
        "issue_resolved": true,
        "follow_up_needed": false,
        "escalation_required": false,
        "customer_satisfied": true,
        "agent_effectiveness": "high/medium/low"
    }},
    "key_phrases": {{
        "customer_pain_points": [],
        "solution_phrases": [],
        "positive_feedback": [],
        "negative_feedback": [],
        "action_items": []
    }},
    "recommendations": {{
        "immediate_actions": [],
        "agent_coaching": [],
        "process_improvements": [],
        "customer_follow_up": []
    }}
}}

Please ensure all scores are between 0.0 and 1.0, and all classifications use the exact values specified above.
"""
        
        return prompt
    
    def call_gemini_api(self, prompt: str, retries: int = 0) -> Optional[Dict]:
        """Call Gemini API with retry logic"""
        try:
            response = self.model.generate_content(prompt)
            
            if response.text:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
                else:
                    logger.warning("No JSON found in response")
                    return None
            else:
                logger.warning("Empty response from Gemini API")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            if retries < self.max_retries:
                time.sleep(self.request_delay * (retries + 1))
                return self.call_gemini_api(prompt, retries + 1)
            return None
            
        except Exception as e:
            logger.error(f"API call error: {e}")
            if retries < self.max_retries:
                time.sleep(self.request_delay * (retries + 1))
                return self.call_gemini_api(prompt, retries + 1)
            return None
    
    def process_single_call(self, row: pd.Series) -> Dict:
        """Process a single call record"""
        try:
            # Preprocess text data
            call_data = {
                'call_id': row.get('Call ID', ''),
                'processed_transcript': self.preprocess_transcript(row.get('Transcript', '')),
                'processed_summary': self.preprocess_summary(row.get('Summary', '')),
                'duration': row.get('Duration', ''),
                'ended_reason': row.get('Ended Reason', ''),
                'customer_name': row.get('Customer Name', ''),
                'assistant_id': row.get('Assistant ID', ''),
                'success_evaluation': row.get('Success Evaluation', '')
            }
            
            # Skip if no transcript or summary
            if not call_data['processed_transcript'] and not call_data['processed_summary']:
                logger.warning(f"No transcript or summary for call {call_data['call_id']}")
                return self.get_default_analysis()
            
            # Create prompt and call API
            prompt = self.create_analysis_prompt(call_data)
            analysis = self.call_gemini_api(prompt)
            
            if analysis:
                return analysis
            else:
                logger.warning(f"Failed to analyze call {call_data['call_id']}")
                return self.get_default_analysis()
                
        except Exception as e:
            logger.error(f"Error processing call: {e}")
            return self.get_default_analysis()
    
    def get_default_analysis(self) -> Dict:
        """Return default analysis structure for failed calls"""
        return {
            "sentiment_analysis": {
                "overall_sentiment": "neutral",
                "overall_sentiment_score": 0.5,
                "customer_sentiment": "neutral",
                "customer_sentiment_score": 0.5,
                "agent_sentiment": "neutral",
                "agent_sentiment_score": 0.5,
                "sentiment_journey": "stable",
                "emotional_indicators": ["neutral"]
            },
            "intent_classification": {
                "primary_intent": "inquiry",
                "secondary_intent": "none",
                "intent_confidence": 0.0,
                "specific_intent": "general_inquiry"
            },
            "entity_extraction": {
                "customer_entities": {
                    "account_numbers": [],
                    "phone_numbers": [],
                    "email_addresses": [],
                    "order_ids": [],
                    "reference_numbers": []
                },
                "business_entities": {
                    "products_mentioned": [],
                    "services_mentioned": [],
                    "competitors_mentioned": [],
                    "departments_mentioned": [],
                    "policies_mentioned": []
                },
                "issue_entities": {
                    "problem_categories": [],
                    "urgency_level": "low",
                    "resolution_status": "unresolved"
                }
            },
            "conversation_analysis": {
                "call_structure_score": 0.5,
                "agent_professionalism": 0.5,
                "customer_satisfaction_predicted": 0.5,
                "communication_clarity": 0.5,
                "problem_resolution_effectiveness": 0.5,
                "call_flow_quality": "fair"
            },
            "quality_metrics": {
                "agent_performance": {
                    "politeness_score": 0.5,
                    "empathy_score": 0.5,
                    "technical_accuracy": 0.5,
                    "response_time_quality": "fair",
                    "follow_up_quality": "fair"
                },
                "customer_experience": {
                    "satisfaction_indicators": [],
                    "frustration_indicators": [],
                    "loyalty_impact": "neutral"
                }
            },
            "business_insights": {
                "churn_risk": "low",
                "upsell_opportunity": "low",
                "training_needs": [],
                "process_improvement_areas": [],
                "cost_efficiency": "medium"
            },
            "topic_analysis": {
                "main_topics": [],
                "subtopics": [],
                "topic_complexity": "low",
                "resolution_complexity": "low"
            },
            "outcome_analysis": {
                "call_success": False,
                "issue_resolved": False,
                "follow_up_needed": False,
                "escalation_required": False,
                "customer_satisfied": False,
                "agent_effectiveness": "medium"
            },
            "key_phrases": {
                "customer_pain_points": [],
                "solution_phrases": [],
                "positive_feedback": [],
                "negative_feedback": [],
                "action_items": []
            },
            "recommendations": {
                "immediate_actions": [],
                "agent_coaching": [],
                "process_improvements": [],
                "customer_follow_up": []
            }
        }
    
    def flatten_analysis(self, analysis: Dict) -> Dict:
        """Flatten nested analysis structure for CSV columns"""
        flattened = {}
        
        # Sentiment Analysis
        sentiment = analysis.get('sentiment_analysis', {})
        flattened.update({
            'overall_sentiment': sentiment.get('overall_sentiment', 'neutral'),
            'overall_sentiment_score': sentiment.get('overall_sentiment_score', 0.5),
            'customer_sentiment': sentiment.get('customer_sentiment', 'neutral'),
            'customer_sentiment_score': sentiment.get('customer_sentiment_score', 0.5),
            'agent_sentiment': sentiment.get('agent_sentiment', 'neutral'),
            'agent_sentiment_score': sentiment.get('agent_sentiment_score', 0.5),
            'sentiment_journey': sentiment.get('sentiment_journey', 'stable'),
            'emotional_indicators': ','.join(sentiment.get('emotional_indicators', []))
        })
        
        # Intent Classification
        intent = analysis.get('intent_classification', {})
        flattened.update({
            'primary_intent': intent.get('primary_intent', 'inquiry'),
            'secondary_intent': intent.get('secondary_intent', 'none'),
            'intent_confidence': intent.get('intent_confidence', 0.0),
            'specific_intent': intent.get('specific_intent', 'general_inquiry')
        })
        
        # Entity Extraction
        entities = analysis.get('entity_extraction', {})
        customer_entities = entities.get('customer_entities', {})
        business_entities = entities.get('business_entities', {})
        issue_entities = entities.get('issue_entities', {})
        
        flattened.update({
            'account_numbers': ','.join(customer_entities.get('account_numbers', [])),
            'products_mentioned': ','.join(business_entities.get('products_mentioned', [])),
            'services_mentioned': ','.join(business_entities.get('services_mentioned', [])),
            'competitors_mentioned': ','.join(business_entities.get('competitors_mentioned', [])),
            'problem_categories': ','.join(issue_entities.get('problem_categories', [])),
            'urgency_level': issue_entities.get('urgency_level', 'low'),
            'resolution_status': issue_entities.get('resolution_status', 'unresolved')
        })
        
        # Conversation Analysis
        conversation = analysis.get('conversation_analysis', {})
        flattened.update({
            'call_structure_score': conversation.get('call_structure_score', 0.5),
            'agent_professionalism': conversation.get('agent_professionalism', 0.5),
            'customer_satisfaction_predicted': conversation.get('customer_satisfaction_predicted', 0.5),
            'communication_clarity': conversation.get('communication_clarity', 0.5),
            'problem_resolution_effectiveness': conversation.get('problem_resolution_effectiveness', 0.5),
            'call_flow_quality': conversation.get('call_flow_quality', 'fair')
        })
        
        # Quality Metrics
        quality = analysis.get('quality_metrics', {})
        agent_perf = quality.get('agent_performance', {})
        customer_exp = quality.get('customer_experience', {})
        
        flattened.update({
            'agent_politeness_score': agent_perf.get('politeness_score', 0.5),
            'agent_empathy_score': agent_perf.get('empathy_score', 0.5),
            'agent_technical_accuracy': agent_perf.get('technical_accuracy', 0.5),
            'agent_response_time_quality': agent_perf.get('response_time_quality', 'fair'),
            'satisfaction_indicators': ','.join(customer_exp.get('satisfaction_indicators', [])),
            'frustration_indicators': ','.join(customer_exp.get('frustration_indicators', [])),
            'loyalty_impact': customer_exp.get('loyalty_impact', 'neutral')
        })
        
        # Business Insights
        business = analysis.get('business_insights', {})
        flattened.update({
            'churn_risk': business.get('churn_risk', 'low'),
            'upsell_opportunity': business.get('upsell_opportunity', 'low'),
            'training_needs': ','.join(business.get('training_needs', [])),
            'process_improvement_areas': ','.join(business.get('process_improvement_areas', [])),
            'cost_efficiency': business.get('cost_efficiency', 'medium')
        })
        
        # Topic Analysis
        topics = analysis.get('topic_analysis', {})
        flattened.update({
            'main_topics': ','.join(topics.get('main_topics', [])),
            'topic_complexity': topics.get('topic_complexity', 'low'),
            'resolution_complexity': topics.get('resolution_complexity', 'low')
        })
        
        # Outcome Analysis
        outcome = analysis.get('outcome_analysis', {})
        flattened.update({
            'call_success': outcome.get('call_success', False),
            'issue_resolved': outcome.get('issue_resolved', False),
            'follow_up_needed': outcome.get('follow_up_needed', False),
            'escalation_required': outcome.get('escalation_required', False),
            'customer_satisfied': outcome.get('customer_satisfied', False),
            'agent_effectiveness': outcome.get('agent_effectiveness', 'medium')
        })
        
        # Key Phrases
        phrases = analysis.get('key_phrases', {})
        flattened.update({
            'customer_pain_points': ','.join(phrases.get('customer_pain_points', [])),
            'solution_phrases': ','.join(phrases.get('solution_phrases', [])),
            'positive_feedback': ','.join(phrases.get('positive_feedback', [])),
            'negative_feedback': ','.join(phrases.get('negative_feedback', []))
        })
        
        # Recommendations
        recommendations = analysis.get('recommendations', {})
        flattened.update({
            'immediate_actions': ','.join(recommendations.get('immediate_actions', [])),
            'agent_coaching': ','.join(recommendations.get('agent_coaching', [])),
            'process_improvements': ','.join(recommendations.get('process_improvements', []))
        })
        
        return flattened
    
    def process_csv(self, input_file: str, output_file: str, batch_size: int = 10):
        """Process the entire CSV file with batch processing"""
        try:
            # Read CSV file
            logger.info(f"Reading CSV file: {input_file}")
            df = pd.read_csv(input_file)
            
            logger.info(f"Processing {len(df)} records")
            
            # Process in batches
            all_results = []
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
                
                batch_results = []
                for idx, row in batch.iterrows():
                    logger.info(f"Processing call {idx + 1}/{len(df)}: {row.get('Call ID', 'N/A')}")
                    
                    # Process single call
                    analysis = self.process_single_call(row)
                    flattened = self.flatten_analysis(analysis)
                    batch_results.append(flattened)
                    
                    # Rate limiting
                    time.sleep(self.request_delay)
                
                all_results.extend(batch_results)
                
                # Save intermediate results
                if i % (batch_size * 5) == 0:  # Save every 5 batches
                    self.save_intermediate_results(df, all_results, output_file, i + batch_size)
            
            # Create new DataFrame with additional columns
            results_df = pd.DataFrame(all_results)
            final_df = pd.concat([df, results_df], axis=1)
            
            # Save final results
            final_df.to_csv(output_file, index=False)
            logger.info(f"Processing complete. Results saved to: {output_file}")
            
            # Generate summary report
            self.generate_summary_report(final_df, output_file.replace('.csv', '_summary.txt'))
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
    
    def save_intermediate_results(self, original_df: pd.DataFrame, results: List[Dict], 
                                output_file: str, processed_count: int):
        """Save intermediate results to prevent data loss"""
        try:
            intermediate_file = output_file.replace('.csv', f'_intermediate_{processed_count}.csv')
            
            # Create DataFrame with processed results so far
            processed_df = original_df.iloc[:len(results)].copy()
            results_df = pd.DataFrame(results)
            combined_df = pd.concat([processed_df, results_df], axis=1)
            
            combined_df.to_csv(intermediate_file, index=False)
            logger.info(f"Intermediate results saved: {intermediate_file}")
            
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
    
    def generate_summary_report(self, df: pd.DataFrame, report_file: str):
        """Generate a summary report of the analysis"""
        try:
            with open(report_file, 'w') as f:
                f.write("CALL CENTER ANALYSIS SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total calls processed: {len(df)}\n\n")
                
                # Sentiment Analysis Summary
                f.write("SENTIMENT ANALYSIS\n")
                f.write("-" * 20 + "\n")
                sentiment_dist = df['overall_sentiment'].value_counts()
                for sentiment, count in sentiment_dist.items():
                    f.write(f"{sentiment.capitalize()}: {count} ({count/len(df)*100:.1f}%)\n")
                f.write(f"Average sentiment score: {df['overall_sentiment_score'].mean():.2f}\n\n")
                
                # Intent Classification Summary
                f.write("INTENT CLASSIFICATION\n")
                f.write("-" * 20 + "\n")
                intent_dist = df['primary_intent'].value_counts()
                for intent, count in intent_dist.items():
                    f.write(f"{intent.capitalize()}: {count} ({count/len(df)*100:.1f}%)\n")
                f.write("\n")
                
                # Quality Metrics Summary
                f.write("QUALITY METRICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average agent professionalism: {df['agent_professionalism'].mean():.2f}\n")
                f.write(f"Average customer satisfaction: {df['customer_satisfaction_predicted'].mean():.2f}\n")
                f.write(f"Average problem resolution effectiveness: {df['problem_resolution_effectiveness'].mean():.2f}\n")
                f.write(f"Calls successfully resolved: {df['call_success'].sum()} ({df['call_success'].sum()/len(df)*100:.1f}%)\n\n")
                
                # Business Insights Summary
                f.write("BUSINESS INSIGHTS\n")
                f.write("-" * 20 + "\n")
                churn_risk_dist = df['churn_risk'].value_counts()
                for risk, count in churn_risk_dist.items():
                    f.write(f"{risk.capitalize()} churn risk: {count} ({count/len(df)*100:.1f}%)\n")
                f.write("\n")
                
                # Top Issues
                f.write("TOP ISSUES\n")
                f.write("-" * 20 + "\n")
                all_categories = []
                for categories in df['problem_categories'].dropna():
                    if categories:
                        all_categories.extend(categories.split(','))
                
                if all_categories:
                    from collections import Counter
                    top_issues = Counter(all_categories).most_common(10)
                    for issue, count in top_issues:
                        f.write(f"{issue.strip()}: {count}\n")
                
                f.write("\n" + "=" * 50 + "\n")
                f.write("End of Report\n")
                
            logger.info(f"Summary report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")

def main():
    """Main function to run the call center processor"""
    
    # Configuration
    INPUT_FILE = "row_data.csv"
    OUTPUT_FILE = "processed_data.csv"
    BATCH_SIZE = 5

    # NEW CONTROL VARIABLES
    PROCESS_ALL = False       # Set to True to process the entire CSV
    NUM_RECORDS = 15         # Only used if PROCESS_ALL is False

    try:
        processor = CallCenterProcessor()

        # Read input CSV once and slice if needed
        df = pd.read_csv(INPUT_FILE)
        if not PROCESS_ALL:
            df = df.head(NUM_RECORDS)
            logger.info(f"Processing only first {NUM_RECORDS} records as PROCESS_ALL is False.")

        # Process the DataFrame directly
        processor.process_dataframe(df, OUTPUT_FILE, BATCH_SIZE)

        print(f"Processing complete! Check {OUTPUT_FILE} for results.")

    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()