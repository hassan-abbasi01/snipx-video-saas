import { useState } from 'react';
import { 
  Search, 
  MessageCircle, 
  Book, 
  HelpCircle, 
  Send, 
  Phone, 
  Mail, 
  Clock,
  ChevronDown,
  ChevronRight,
  Bug,
  FileText,
  Video,
  Scissors,
  Type,
  Volume2,
  Sparkles
} from 'lucide-react';
import toast from 'react-hot-toast';
import { ApiService } from '../services/api';
import { useAuth } from '../contexts/AuthContext';

interface FAQ {
  id: string;
  question: string;
  answer: string;
  category: string;
}

const Help = () => {
  const { isAuthenticated, user } = useAuth();
  const [activeTab, setActiveTab] = useState('faq');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [expandedFAQ, setExpandedFAQ] = useState<string | null>(null);
  const [supportForm, setSupportForm] = useState({
    name: user?.firstName && user?.lastName ? `${user.firstName} ${user.lastName}` : '',
    email: user?.email || '',
    subject: '',
    description: '',
    priority: 'medium',
    type: 'bug'
  });
  const [isSubmittingTicket, setIsSubmittingTicket] = useState(false);

  const faqs: FAQ[] = [
    {
      id: '1',
      question: 'How do I upload a video to SnipX?',
      answer: 'To upload a video, go to the Editor page and either drag & drop your video file or click the "Select Video" button. We support MP4, MOV, AVI, and MKV formats up to 500MB.',
      category: 'upload'
    },
    {
      id: '2',
      question: 'What video formats are supported?',
      answer: 'SnipX supports MP4, MOV, AVI, MKV, and WMV formats. For best results, we recommend using MP4 format.',
      category: 'upload'
    },
    {
      id: '3',
      question: 'How does the automatic subtitle generation work?',
      answer: 'Our AI analyzes the audio in your video and generates accurate subtitles. You can choose from multiple languages including English, Urdu, Spanish, French, and German.',
      category: 'subtitles'
    },
    {
      id: '4',
      question: 'Can I edit the generated subtitles?',
      answer: 'Yes! After subtitles are generated, you can edit them directly in the editor. You can modify text, timing, and styling.',
      category: 'subtitles'
    },
    {
      id: '5',
      question: 'What is the silence cutting feature?',
      answer: 'The silence cutting feature automatically detects and removes silent parts from your video, making it more engaging and reducing file size.',
      category: 'editing'
    },
    {
      id: '6',
      question: 'How long does video processing take?',
      answer: 'Processing time depends on video length and selected features. Typically, it takes 1-3 minutes per minute of video content.',
      category: 'processing'
    },
    {
      id: '7',
      question: 'Is my video data secure?',
      answer: 'Yes, we use enterprise-grade encryption and secure cloud storage. Your videos are private and only accessible to you.',
      category: 'security'
    },
    {
      id: '8',
      question: 'Can I download my processed videos?',
      answer: 'Absolutely! Once processing is complete, you can download your enhanced video in various quality settings.',
      category: 'download'
    }
  ];

  const categories = [
    { id: 'all', label: 'All Categories' },
    { id: 'upload', label: 'Upload & Import' },
    { id: 'editing', label: 'Video Editing' },
    { id: 'subtitles', label: 'Subtitles' },
    { id: 'processing', label: 'Processing' },
    { id: 'download', label: 'Export & Download' },
    { id: 'security', label: 'Security' }
  ];

  const tutorials = [
    {
      id: '1',
      title: 'Getting Started with SnipX',
      description: 'Learn the basics of uploading and processing your first video',
      duration: '5 min',
      icon: <Video className="text-blue-500" size={24} />
    },
    {
      id: '2',
      title: 'Advanced Subtitle Editing',
      description: 'Master subtitle generation and customization features',
      duration: '8 min',
      icon: <Type className="text-green-500" size={24} />
    },
    {
      id: '3',
      title: 'Audio Enhancement Techniques',
      description: 'Improve your video audio quality with AI-powered tools',
      duration: '6 min',
      icon: <Volume2 className="text-purple-500" size={24} />
    },
    {
      id: '4',
      title: 'Automated Video Cutting',
      description: 'Use AI to automatically remove silences and improve pacing',
      duration: '7 min',
      icon: <Scissors className="text-red-500" size={24} />
    }
  ];

  const filteredFAQs = faqs.filter(faq => {
    const matchesSearch = faq.question.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         faq.answer.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || faq.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const handleSupportSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!isAuthenticated) {
      toast.error('Please log in to submit a support ticket.');
      return;
    }
    
    setIsSubmittingTicket(true);

    try {
      // Submit as authenticated user
      await ApiService.submitSupportTicket(supportForm);
      toast.success('Support ticket submitted successfully! We\'ll get back to you within 24 hours.');
      console.log('Ticket saved to MongoDB via authenticated API');

      // Reset form
      setSupportForm({
        name: user?.firstName && user?.lastName ? `${user.firstName} ${user.lastName}` : '',
        email: user?.email || '',
        subject: '',
        description: '',
        priority: 'medium',
        type: 'bug'
      });
    } catch (error) {
      console.error('Error submitting support ticket:', error);
      toast.error('Failed to submit support ticket. Please try again.');
    } finally {
      setIsSubmittingTicket(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-purple-50 py-8 relative overflow-hidden">
      {/* 3D Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Floating 3D Spheres */}
        <div className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-br from-purple-400/20 to-pink-400/20 rounded-full blur-xl animate-float-3d transform-gpu" 
             style={{ transform: 'translateZ(0) rotateX(45deg) rotateY(45deg)' }} />
        <div className="absolute top-40 right-20 w-24 h-24 bg-gradient-to-br from-blue-400/20 to-cyan-400/20 rounded-full blur-lg animate-float-3d-delayed transform-gpu"
             style={{ transform: 'translateZ(0) rotateX(-30deg) rotateY(60deg)' }} />
        <div className="absolute bottom-32 left-1/4 w-40 h-40 bg-gradient-to-br from-green-400/15 to-teal-400/15 rounded-full blur-2xl animate-pulse-3d transform-gpu"
             style={{ transform: 'translateZ(0) rotateX(60deg) rotateY(-45deg)' }} />
        
        {/* 3D Geometric Shapes */}
        <div className="absolute top-1/3 right-1/4 w-16 h-16 bg-gradient-to-br from-orange-400/30 to-red-400/30 transform rotate-45 animate-spin-3d blur-sm" />
        <div className="absolute bottom-1/4 right-1/3 w-12 h-12 bg-gradient-to-br from-indigo-400/30 to-purple-400/30 transform rotate-12 animate-bounce-3d blur-sm" />
        
        {/* Floating Sparkles */}
        <div className="absolute top-1/4 left-1/3 animate-sparkle-3d">
          <Sparkles className="text-purple-400/40 w-6 h-6 transform-gpu" style={{ transform: 'rotateZ(45deg)' }} />
        </div>
        <div className="absolute top-2/3 right-1/2 animate-sparkle-3d-delayed">
          <Sparkles className="text-pink-400/40 w-4 h-4 transform-gpu" style={{ transform: 'rotateZ(-30deg)' }} />
        </div>
      </div>

      <div className="container mx-auto px-4 max-w-6xl relative z-10">
        {/* Header with 3D Effects */}
        <div className="text-center mb-8 animate-slide-up-3d">
          <div className="inline-flex items-center px-6 py-3 bg-white/80 backdrop-blur-md rounded-full shadow-lg border border-white/20 mb-6 transform hover:scale-105 transition-all duration-300 hover:shadow-xl">
            <HelpCircle className="text-purple-600 mr-3 animate-pulse" size={24} />
            <span className="text-purple-700 font-medium">Help & Support Center</span>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent animate-text-shimmer">
            Help & Support
          </h1>
          <p className="text-xl text-gray-600 animate-fade-in-up-3d">
            Find answers, get help, and learn how to make the most of SnipX
          </p>
        </div>

        {/* Navigation Tabs with 3D Effects */}
        <div className="bg-white/90 backdrop-blur-md rounded-2xl shadow-xl mb-6 border border-white/20 animate-slide-in-3d">
          <nav className="flex space-x-8 px-6 overflow-x-auto">
            {[
              { id: 'faq', label: 'FAQ', icon: HelpCircle },
              { id: 'tutorials', label: 'Tutorials', icon: Book },
              { id: 'support', label: 'Support Ticket', icon: Bug }
            ].map((tab, index) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center py-4 px-2 border-b-2 font-medium text-sm transition-all duration-300 transform hover:scale-105 hover:-translate-y-1 ${
                  activeTab === tab.id
                    ? 'border-purple-500 text-purple-600 shadow-lg'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <tab.icon size={16} className="mr-2 animate-pulse" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content with 3D Animations */}
        <div className="bg-white/90 backdrop-blur-md rounded-2xl shadow-xl p-6 border border-white/20 animate-content-reveal-3d">
          {activeTab === 'faq' && (
            <div className="space-y-6">
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1 relative group">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 group-hover:text-purple-500 transition-colors" size={16} />
                  <input
                    type="text"
                    placeholder="Search FAQs..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white/80 backdrop-blur-sm transition-all duration-300 hover:shadow-lg"
                  />
                </div>
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white/80 backdrop-blur-sm transition-all duration-300 hover:shadow-lg"
                >
                  {categories.map(category => (
                    <option key={category.id} value={category.id}>
                      {category.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-4">
                {filteredFAQs.map((faq, index) => (
                  <div 
                    key={faq.id} 
                    className="border border-gray-200 rounded-xl bg-white/80 backdrop-blur-sm shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 hover:scale-[1.02] animate-slide-in-stagger-3d"
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <button
                      onClick={() => setExpandedFAQ(expandedFAQ === faq.id ? null : faq.id)}
                      className="w-full flex items-center justify-between p-6 text-left hover:bg-purple-50/50 rounded-xl transition-all duration-300"
                    >
                      <span className="font-medium text-gray-900 pr-4">{faq.question}</span>
                      <div className="transform transition-transform duration-300" style={{ transform: expandedFAQ === faq.id ? 'rotateZ(180deg)' : 'rotateZ(0deg)' }}>
                        <ChevronDown size={20} className="text-gray-500" />
                      </div>
                    </button>
                    {expandedFAQ === faq.id && (
                      <div className="px-6 pb-6 text-gray-600 animate-expand-3d">
                        {faq.answer}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {filteredFAQs.length === 0 && (
                <div className="text-center py-12 animate-fade-in-3d">
                  <HelpCircle className="mx-auto text-gray-400 mb-4 animate-bounce-3d" size={48} />
                  <p className="text-gray-500">No FAQs found matching your search.</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'tutorials' && (
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold text-gray-900 animate-slide-in-3d">Video Tutorials</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {tutorials.map((tutorial, index) => (
                  <div 
                    key={tutorial.id} 
                    className="border border-gray-200 rounded-xl p-6 hover:shadow-xl transition-all duration-500 bg-white/80 backdrop-blur-sm transform hover:-translate-y-2 hover:scale-105 animate-card-float-3d"
                    style={{ animationDelay: `${index * 150}ms` }}
                  >
                    <div className="flex items-start">
                      <div className="flex-shrink-0 mr-4 transform hover:scale-110 transition-transform duration-300">
                        {tutorial.icon}
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-medium text-gray-900 mb-2 hover:text-purple-600 transition-colors">
                          {tutorial.title}
                        </h3>
                        <p className="text-gray-600 mb-3">{tutorial.description}</p>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-500 flex items-center">
                            <Clock size={14} className="mr-1" />
                            {tutorial.duration}
                          </span>
                          <button className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-4 py-2 rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-300 transform hover:scale-105 hover:shadow-lg">
                            Watch Tutorial
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-xl p-6 animate-slide-up-3d">
                <h3 className="text-lg font-medium text-purple-900 mb-2">Need More Help?</h3>
                <p className="text-purple-700 mb-4">
                  Can't find what you're looking for? Check out our comprehensive documentation or contact our support team.
                </p>
                <div className="flex space-x-4">
                  <button className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-6 py-3 rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-300 transform hover:scale-105 hover:shadow-lg">
                    View Documentation
                  </button>
                  <button className="border-2 border-purple-600 text-purple-600 px-6 py-3 rounded-lg hover:bg-purple-50 transition-all duration-300 transform hover:scale-105">
                    Contact Support
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'support' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-semibold text-gray-900 animate-slide-in-3d">Submit Support Ticket</h2>
                <div className="flex items-center space-x-4 text-sm text-gray-600">
                  <div className="flex items-center animate-float-3d">
                    <Mail size={16} className="mr-1" />
                    support@snipx.com
                  </div>
                  <div className="flex items-center animate-float-3d-delayed">
                    <Phone size={16} className="mr-1" />
                    +1 (555) 123-4567
                  </div>
                </div>
              </div>

              {!isAuthenticated ? (
                <div className="bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-200 rounded-xl p-6 text-center animate-slide-up-3d">
                  <div className="flex justify-center mb-4">
                    <Bug className="h-12 w-12 text-blue-600 animate-bounce-3d" />
                  </div>
                  <h3 className="text-lg font-semibold text-blue-900 mb-2">
                    Login Required
                  </h3>
                  <p className="text-blue-700 mb-4">
                    Please log in to submit a support ticket. This helps us track your issues and provide better support.
                  </p>
                  <button
                    onClick={() => window.location.href = '/login'}
                    className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white px-6 py-3 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 hover:shadow-lg"
                  >
                    Go to Login
                  </button>
                </div>
              ) : (
                <form onSubmit={handleSupportSubmit} className="space-y-6 animate-form-reveal-3d">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="animate-slide-in-left-3d">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Full Name
                      </label>
                      <input
                        type="text"
                        value={supportForm.name}
                        onChange={(e) => setSupportForm({...supportForm, name: e.target.value})}
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white/80 backdrop-blur-sm transition-all duration-300 hover:shadow-lg"
                        required
                      />
                    </div>

                    <div className="animate-slide-in-right-3d">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Email Address
                      </label>
                      <input
                        type="email"
                        value={supportForm.email}
                        onChange={(e) => setSupportForm({...supportForm, email: e.target.value})}
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white/80 backdrop-blur-sm transition-all duration-300 hover:shadow-lg"
                        required
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="animate-slide-in-left-3d" style={{ animationDelay: '200ms' }}>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Issue Type
                      </label>
                      <select
                        value={supportForm.type}
                        onChange={(e) => setSupportForm({...supportForm, type: e.target.value})}
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white/80 backdrop-blur-sm transition-all duration-300 hover:shadow-lg"
                      >
                        <option value="bug">Bug Report</option>
                        <option value="feature">Feature Request</option>
                        <option value="account">Account Issue</option>
                        <option value="billing">Billing Question</option>
                        <option value="other">Other</option>
                      </select>
                    </div>

                    <div className="animate-slide-in-right-3d" style={{ animationDelay: '200ms' }}>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Priority
                      </label>
                      <select
                        value={supportForm.priority}
                        onChange={(e) => setSupportForm({...supportForm, priority: e.target.value})}
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white/80 backdrop-blur-sm transition-all duration-300 hover:shadow-lg"
                      >
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                        <option value="urgent">Urgent</option>
                      </select>
                    </div>
                  </div>

                  <div className="animate-slide-up-3d" style={{ animationDelay: '300ms' }}>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Subject
                    </label>
                    <input
                      type="text"
                      value={supportForm.subject}
                      onChange={(e) => setSupportForm({...supportForm, subject: e.target.value})}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white/80 backdrop-blur-sm transition-all duration-300 hover:shadow-lg"
                      required
                    />
                  </div>

                  <div className="animate-slide-up-3d" style={{ animationDelay: '400ms' }}>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Description
                    </label>
                    <textarea
                      value={supportForm.description}
                      onChange={(e) => setSupportForm({...supportForm, description: e.target.value})}
                      rows={6}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white/80 backdrop-blur-sm transition-all duration-300 hover:shadow-lg resize-none"
                      placeholder="Please provide as much detail as possible about your issue..."
                      required
                    />
                  </div>

                  <div className="flex justify-end animate-slide-up-3d" style={{ animationDelay: '500ms' }}>
                    <button
                      type="submit"
                      disabled={isSubmittingTicket}
                      className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 py-3 rounded-xl hover:from-purple-700 hover:to-pink-700 transition-all duration-300 flex items-center disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 hover:shadow-lg"
                    >
                      <FileText size={16} className="mr-2" />
                      {isSubmittingTicket ? 'Submitting...' : 'Submit Ticket'}
                    </button>
                  </div>
                </form>
              )}

              <div className="bg-gradient-to-r from-gray-50 to-slate-50 border border-gray-200 rounded-xl p-6 animate-slide-up-3d">
                <h3 className="text-sm font-medium text-gray-900 mb-4">Response Times</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  {[
                    { priority: 'Urgent', time: '2-4 hours', color: 'red' },
                    { priority: 'High', time: '4-8 hours', color: 'orange' },
                    { priority: 'Medium', time: '12-24 hours', color: 'yellow' },
                    { priority: 'Low', time: '24-48 hours', color: 'green' }
                  ].map((item, index) => (
                    <div 
                      key={item.priority}
                      className="animate-bounce-in-3d"
                      style={{ animationDelay: `${index * 100}ms` }}
                    >
                      <span className={`font-medium text-${item.color}-600`}>{item.priority}:</span>
                      <p className="text-gray-600">{item.time}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Help;