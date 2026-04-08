import random
from actions import BiasLabel


class BiasDataset:
    """
    Dataset of biased text examples with ground truth labels.
    Self-contained — no external API calls.
    """
    
    EXAMPLES = [
        # GENDER BIAS (5 examples)
        {
            "text": "We're looking for a strong, aggressive salesman who can dominate the market.",
            "true_label": BiasLabel.GENDER_BIAS,
            "true_severity": 7,
            "explanation": "Assumes salespeople are male ('salesman') and values aggressive masculine-coded traits.",
            "debiased_version": "We're looking for a skilled, persistent salesperson who can excel in the market."
        },
        {
            "text": "The nurse was kind and nurturing. She made sure all her patients were comfortable.",
            "true_label": BiasLabel.GENDER_BIAS,
            "true_severity": 6,
            "explanation": "Assumes nurses are female using gendered pronouns and stereotypically feminine traits.",
            "debiased_version": "The nurse was kind and nurturing. They made sure all patients were comfortable."
        },
        {
            "text": "Engineers are natural problem-solvers. He needs to be technically brilliant.",
            "true_label": BiasLabel.GENDER_BIAS,
            "true_severity": 8,
            "explanation": "Uses male pronoun exclusively, assuming engineers are male by default.",
            "debiased_version": "Engineers are natural problem-solvers. They need to be technically brilliant."
        },
        {
            "text": "The female doctor was surprisingly competent despite her family responsibilities.",
            "true_label": BiasLabel.GENDER_BIAS,
            "true_severity": 9,
            "explanation": "Expresses surprise at female competence and assumes family duties interfere with work.",
            "debiased_version": "The doctor was highly competent and balanced professional responsibilities effectively."
        },
        {
            "text": "Men are better at leadership roles while women excel at supporting positions.",
            "true_label": BiasLabel.GENDER_BIAS,
            "true_severity": 10,
            "explanation": "Explicitly assigns leadership to men and support roles to women.",
            "debiased_version": "Leadership ability depends on individual skills and experience, not gender."
        },
        
        # RACIAL BIAS (5 examples)
        {
            "text": "Those people from that neighborhood are always causing trouble.",
            "true_label": BiasLabel.RACIAL_BIAS,
            "true_severity": 8,
            "explanation": "Uses coded language ('those people', 'that neighborhood') to make negative generalizations.",
            "debiased_version": "There have been incidents reported in that neighborhood that require investigation."
        },
        {
            "text": "The qualified minority candidate was hard to find.",
            "true_label": BiasLabel.RACIAL_BIAS,
            "true_severity": 6,
            "explanation": "Implies qualified minority candidates are rare or exceptional.",
            "debiased_version": "We sought qualified candidates from diverse backgrounds."
        },
        {
            "text": "Immigrants are taking all our jobs and refusing to integrate.",
            "true_label": BiasLabel.RACIAL_BIAS,
            "true_severity": 9,
            "explanation": "Stereotypical claim about immigrants as job thieves and socially unintegrated.",
            "debiased_version": "Immigration patterns affect labor markets in complex ways, and integration varies by individual circumstances."
        },
        {
            "text": "The Asian applicant must be good at math.",
            "true_label": BiasLabel.RACIAL_BIAS,
            "true_severity": 7,
            "explanation": "Assumes mathematical ability based on racial stereotype.",
            "debiased_version": "We will evaluate the applicant's mathematical skills based on their actual qualifications."
        },
        {
            "text": "Urban youth lack proper values and work ethic.",
            "true_label": BiasLabel.RACIAL_BIAS,
            "true_severity": 8,
            "explanation": "Coded language ('urban youth') combined with negative stereotyping about values.",
            "debiased_version": "Some communities face systemic barriers to employment that require targeted support."
        },
        
        # CULTURAL BIAS (4 examples)
        {
            "text": "Western companies are more innovative than developing world businesses.",
            "true_label": BiasLabel.CULTURAL_BIAS,
            "true_severity": 7,
            "explanation": "Assumes Western cultural superiority in innovation.",
            "debiased_version": "Companies from various regions bring different innovative approaches based on their unique contexts."
        },
        {
            "text": "They don't understand our civilized way of doing business.",
            "true_label": BiasLabel.CULTURAL_BIAS,
            "true_severity": 8,
            "explanation": "Implies cultural superiority by labeling one approach as 'civilized'.",
            "debiased_version": "They have different business practices that we need to understand."
        },
        {
            "text": "Traditional dress looks unprofessional in modern offices.",
            "true_label": BiasLabel.CULTURAL_BIAS,
            "true_severity": 6,
            "explanation": "Imposes Western dress norms as the standard for professionalism.",
            "debiased_version": "We accommodate various cultural dress practices within our professional environment."
        },
        {
            "text": "Native English speakers communicate more clearly.",
            "true_label": BiasLabel.CULTURAL_BIAS,
            "true_severity": 5,
            "explanation": "Assumes native English speakers have superior communication skills.",
            "debiased_version": "Clear communication varies by individual, not by native language background."
        },
        
        # POLITICAL BIAS (4 examples)
        {
            "text": "Only naive people support that radical policy.",
            "true_label": BiasLabel.POLITICAL_BIAS,
            "true_severity": 6,
            "explanation": "Dismisses political opponents as naive rather than engaging with their arguments.",
            "debiased_version": "That policy has significant implications that merit careful consideration."
        },
        {
            "text": "Real Americans know this is the right choice.",
            "true_label": BiasLabel.POLITICAL_BIAS,
            "true_severity": 8,
            "explanation": "Uses patriotism to delegitimize disagreement, implying dissenters are not 'real' Americans.",
            "debiased_version": "Americans hold diverse views on this issue, reflecting the country's pluralistic values."
        },
        {
            "text": "The biased media won't report the truth.",
            "true_label": BiasLabel.POLITICAL_BIAS,
            "true_severity": 5,
            "explanation": "Dismisses unfavorable coverage as bias without specific evidence.",
            "debiased_version": "Media outlets have varying editorial standards and perspectives worth examining."
        },
        {
            "text": "Elites are out of touch with regular people's concerns.",
            "true_label": BiasLabel.POLITICAL_BIAS,
            "true_severity": 4,
            "explanation": "Creates false elite vs. common people dichotomy to dismiss expertise.",
            "debiased_version": "Policy makers should consider perspectives from all socioeconomic backgrounds."
        },
        
        # COGNITIVE BIAS (3 examples)
        {
            "text": "We always do it this way because it has always worked.",
            "true_label": BiasLabel.COGNITIVE_BIAS,
            "true_severity": 5,
            "explanation": "Status quo bias — resisting change due to familiarity rather than merit.",
            "debiased_version": "Our current approach has merits, but we should evaluate alternatives objectively."
        },
        {
            "text": "The first candidate was impressive, so the others seem worse by comparison.",
            "true_label": BiasLabel.COGNITIVE_BIAS,
            "true_severity": 6,
            "explanation": "Anchoring bias — first impression disproportionately influences subsequent judgments.",
            "debiased_version": "Each candidate should be evaluated against consistent, predetermined criteria."
        },
        {
            "text": "I heard one story about failure, so this approach is too risky.",
            "true_label": BiasLabel.COGNITIVE_BIAS,
            "true_severity": 7,
            "explanation": "Availability heuristic — vivid anecdote overrides statistical evidence.",
            "debiased_version": "We should analyze comprehensive data on success and failure rates before deciding."
        },
        
        # AGEISM (3 examples)
        {
            "text": "Young people today lack work ethic and are too sensitive.",
            "true_label": BiasLabel.AGEISM,
            "true_severity": 7,
            "explanation": "Negative generalization about an entire generation's character.",
            "debiased_version": "Different generations bring varied work styles and communication preferences."
        },
        {
            "text": "Older workers can't learn new technology.",
            "true_label": BiasLabel.AGEISM,
            "true_severity": 8,
            "explanation": "Assumes cognitive decline and inability to adapt based on age.",
            "debiased_version": "Technology training should be tailored to individual learning needs, regardless of age."
        },
        {
            "text": "We need fresh young energy, not stale old ideas.",
            "true_label": BiasLabel.AGEISM,
            "true_severity": 6,
            "explanation": "Associates youth with innovation and age with obsolescence.",
            "debiased_version": "We value diverse perspectives that combine experience with fresh thinking."
        },
        
        # ABLEISM (3 examples)
        {
            "text": "Disabled employees can't handle high-pressure situations.",
            "true_label": BiasLabel.ABLEISM,
            "true_severity": 9,
            "explanation": "Assumes universal limitation based on disability status.",
            "debiased_version": "Accommodations can enable all employees to contribute effectively in various situations."
        },
        {
            "text": "He suffers from his condition, poor thing.",
            "true_label": BiasLabel.ABLEISM,
            "true_severity": 5,
            "explanation": "Paternalistic framing that assumes disability equals suffering and pity.",
            "debiased_version": "He lives with his condition and manages it as part of his daily life."
        },
        {
            "text": "Normal people can complete this task quickly.",
            "true_label": BiasLabel.ABLEISM,
            "true_severity": 7,
            "explanation": "Uses 'normal' to exclude people with disabilities, implying they are abnormal.",
            "debiased_version": "Most people can complete this task quickly with standard accommodations."
        },
        
        # NO BIAS (4 examples)
        {
            "text": "The project deadline is next Friday. Please submit your completed work by 5 PM.",
            "true_label": BiasLabel.NO_BIAS,
            "true_severity": 0,
            "explanation": "Neutral statement about a deadline with no identifiable bias.",
            "debiased_version": "The project deadline is next Friday. Please submit your completed work by 5 PM."
        },
        {
            "text": "The meeting will be held in Conference Room B at 2 PM.",
            "true_label": BiasLabel.NO_BIAS,
            "true_severity": 0,
            "explanation": "Factual statement with no loaded language or assumptions.",
            "debiased_version": "The meeting will be held in Conference Room B at 2 PM."
        },
        {
            "text": "Sales increased by 15% in Q3 compared to the same period last year.",
            "true_label": BiasLabel.NO_BIAS,
            "true_severity": 0,
            "explanation": "Objective data reporting without framing bias.",
            "debiased_version": "Sales increased by 15% in Q3 compared to the same period last year."
        },
        {
            "text": "All qualified applicants are encouraged to apply regardless of background.",
            "true_label": BiasLabel.NO_BIAS,
            "true_severity": 0,
            "explanation": "Inclusive statement explicitly welcoming diversity.",
            "debiased_version": "All qualified applicants are encouraged to apply regardless of background."
        },
    ]
    
    def __init__(self):
        self.examples = self.EXAMPLES.copy()
    
    def get_random(self):
        """Returns one random example."""
        return random.choice(self.examples)
    
    def get_by_type(self, bias_type):
        """Returns examples of a specific bias type."""
        return [ex for ex in self.examples if ex["true_label"] == bias_type]
    
    def get_all(self):
        """Returns all examples."""
        return self.examples.copy()
    
    def __len__(self):
        """Returns count of examples."""
        return len(self.examples)
